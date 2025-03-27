use std::{hash::{DefaultHasher, Hash, Hasher}};

//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.

type HashValue = u64;

/// Merkle tree struct definition
/// Root is distinguished for easier access and code legibility
/// layers represents the layers from the leaves up to the layer before the root, with the leaves array being the first and the topmost layer the last
/// unique_elements represents the amount of non-repeated elements in the leaves
struct MerkleTree {
    root: HashValue,
    layers: Vec<Vec<HashValue>>,
    unique_elements: usize
}

/// Takes an element and returns its hash
/// Input: a reference to a hashable element which we want the hash of
/// Output: the element's hash
fn hash<T: Hash> (val: &T) -> HashValue {
    //For now I use the default hasher
    //TODO: Hash algorithm to use could be made parametric
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    let hash = hasher.finish();
    hash
}

/// Calculates the closest power of 2 that's larger than the input
/// Auxiliary function used for some operations over the tree
fn closest_bigger_power_of2 (num: f64) -> u32 {
    let exp = f64::log2(num).ceil();
    f64::powf(2.0, exp) as u32
}

impl MerkleTree {
    /// Merkle tree creation function
    /// Input: A vector of data
    /// Output: a Merkle tree with the hashes of said data as its leaves
    pub fn new<T: Hash> (data: Vec<T>) -> MerkleTree {

        // Hash every element
        let mut hashes: Vec<HashValue> = data.iter().map(| elem | hash(elem)).collect();

        // The amount of leaves always has to be a power of two so the tree remains balanced
        // We achieve this by copying the last element the necessary amount of times to make the size of the leaves array a power of 23
        let last_elem = *hashes.last().unwrap();
        let repetitions: u64 = closest_bigger_power_of2(data.len() as f64) as u64 - data.len() as u64;
        let last_elem_position = hashes.len();
        if repetitions > 0 {
            let mut last_elem_repetitions: Vec<HashValue> = vec![last_elem; repetitions as usize];
            hashes.append(&mut last_elem_repetitions);
        }

        let mut tree = MerkleTree { root: hashes[0].clone(), layers: MerkleTree::create_layers(hashes), unique_elements: last_elem_position };

        if last_elem_position > 1 {
            tree.root = hash(&((*tree.layers.last().unwrap())[0] as u128 + (*tree.layers.last().unwrap())[1] as u128));
        } 
        tree
    }

    /// Function that dynamically adds an element to the merkle tree
    /// Input: an element of a hashable type
    /// Output: none, the tree is modified dynamically
    pub fn add_element<T: Hash>(&mut self, elem: T) {
        let leaves_count = self.layers[0].len();

        // I check if the amount of unique elements is already a power of 2, or if I had to fill with repeats
        // If its a number of two, I create a new subtree (as a vector of vectors) and append the elements of the new subtree to the corresponding layer vector
        // If not, I replace the first repeated instance of an element with the new element
        let repeats_present = leaves_count > self.unique_elements;

        if !repeats_present {
            let mut new_layers = MerkleTree::create_layers(vec![hash(&elem); leaves_count]);
            for (layer_old, layer_new) in self.layers.iter_mut().zip(new_layers.iter_mut()) {
                layer_old.append(layer_new);
            }
            let hash_pre_root_left = hash(&(self.layers.last().unwrap()[0] as u128 + self.layers.last().unwrap()[1] as u128 ));
            let hash_pre_root_right = hash(&(self.layers.last().unwrap()[2] as u128 + self.layers.last().unwrap()[3] as u128 ));

            let last_layer_before_root = vec![hash_pre_root_left, hash_pre_root_right];
            self.layers.push(last_layer_before_root.clone());

            self.root = hash(&(last_layer_before_root[0] as u128 + last_layer_before_root[1] as u128));
        } else {

            //I take the index of the first element that's repeated and change it for the hash of the new element
            let mut index_to_be_updated = self.unique_elements;
            self.layers[0][index_to_be_updated] = hash(&elem);

            // Then I have to go through every upper layer updating the hashes
            // If I update the node of index i in layer j, then in layer j+1 (its upper layer) its parent will be i/2
            // Its brother will be i if its odd, and i+1 if its even. Because I do an integer division beforehand, i can always just add one
            index_to_be_updated /= 2;
            for i in 1..self.layers.len() {
                let new_hash = hash(&(self.layers[i-1][index_to_be_updated*2] as u128 + self.layers[i-1][index_to_be_updated*2+1] as u128));
                self.layers[i][index_to_be_updated] = new_hash;
            }
            self.root = hash(&(self.layers.last().unwrap()[0] as u128 + self.layers.last().unwrap()[1] as u128));
        }

        self.unique_elements += 1;
    }

    /// Auxiliary function to create a sort of subtree with no root
    /// Input: a vector of hashes
    /// Output: a subtree with those hashes as its leaves
    fn create_layers(hashes: Vec<HashValue>) -> Vec<Vec<HashValue>> {
        let mut current_layer = hashes.clone();

        let layers_amount = f64::log2(hashes.len() as f64) as usize;
        let mut layers: Vec<Vec<HashValue>> = Vec::new();

        // I go through each of the layers, starting with the leaves, and create the layer on top of it by adding the hashes of the two elements below, adding them, and hashing the sum
        // I need to put them in the intermediate layers array in reverse order, so the smallest layer is at the end of the array

        for _i in 0..layers_amount {
            layers.push(current_layer.clone());
            let mut next_layer = Vec::new();

            for j in 0..(current_layer.len()/2) {
                let new_hash = hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128));
                next_layer.push(new_hash);
            } 
            current_layer = next_layer.clone();
        }
        layers
    }

    /// Creates a proof that an element is in the merkle tree
    /// The root is all the hashes needed to reconstruct the root hash bottom up, while only having the hash of the element provided
    /// Input: the element for which we want to create a proof
    /// Output: the proof vector, or None if the element isn't present
    pub fn generate_proof<T: Hash>(&self, elem: T) -> Option<Vec<HashValue>> {
        // I look for the position of the element in the leaves array and pattern match on it
        let element_index = self.find_element(elem);
        if let Some(pos) = element_index {
            let mut current_proof_elem_index = pos;
            let mut proof: Vec<HashValue> = Vec::new();
            //I go through the layers. Left children nodes have even indexes, and right children have odd ones
            //I always want the node's brother's hash, since its what I need with the one I currently have to reach the root
            //Hence if the index is even, I get the brother by adding one, and if it's odd, I get the brother by subtracting one
            for layer in self.layers.iter() {
                if current_proof_elem_index % 2 == 0 {
                    proof.push(layer[current_proof_elem_index + 1]);
                } else { proof.push(layer[current_proof_elem_index - 1]); }
                current_proof_elem_index /= 2;
            }
            Some(proof)
        } else { None }
        //I return None if the element wasn't in the tree
    }

    /// Auxiliary function to find the index of the hash of an element in the leaves array
    /// Input: an element
    /// Output: the index of the element if its hash is in the tree, or None if it can't be found
    fn find_element<T: Hash>(&self, elem:T) -> Option<usize> {
        let index = self.layers[0].iter().position(|&r| r == hash(&elem));
        index
    }

    /// Function that takes a proof of the presence of the hash of an element on the tree, and the element, and returns whether the proof can be validated or not
    /// Input: a proof as an array of hashes going from bottom to top of the tree, and an element for which we want to validate said proof
    /// Output: a bool representing whether the proof is valid
    pub fn validate_proof<T: Hash>(&self, proof: Vec<HashValue>, elem: T) -> bool {

        let mut current_hash = hash(&elem);

        //I work from the leaves up generating the hashes which I'd need to eventually get the root hash
        for i in 0..self.layers.len() {
            current_hash = hash(&(current_hash as u128 + proof[i] as u128));
        }
        self.root == current_hash
    }
}


fn main() {
}

#[cfg(test)]
mod tests {

    use super::*;
    
    #[test]
    fn merkle_tree_generation_works_for_one_int() {
        let test_int = 42;
        let merkle_tree_42 = MerkleTree::new(vec![test_int]);
        assert_eq!(merkle_tree_42.root, hash(&test_int));
        assert_eq!(merkle_tree_42.unique_elements, 1);
    }

    #[test]
    fn merkle_tree_generation_works_for_two_ints() {
        let test_ints = vec![4,2];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let mut current_layer: Vec<HashValue> = test_ints.iter().map(|x| hash(x)).collect();

        for i in 0..merkle_tree.layers.len() {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));
            }
            current_layer = next_layer;
        }

        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 2);
    }

    #[test]
    fn merkle_tree_generation_works_for_four_ints() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let mut current_layer: Vec<HashValue> = test_ints.iter().map(|x| hash(x)).collect();

        for i in 0..merkle_tree.layers.len() {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));
            }
            current_layer = next_layer;
        }

        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 4);
    }

    #[test]
    fn merkle_tree_generation_non_power_of_two_amount_of_ints() {
        let mut test_ints = vec![4,8,15,16,13,42];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let mut repeats = vec![42,42];

        test_ints.append(&mut repeats);

        let mut current_layer: Vec<HashValue> = test_ints.iter().map(|x| hash(x)).collect();

        for i in 0..merkle_tree.layers.len() {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));
            }
            current_layer = next_layer;
        }
        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 6);
    }

    #[test]
    fn add_element_works_for_trees_with_power_of_2_unique_elements() {
        let mut test_ints = vec![4,8,15,16];
        let mut merkle_tree = MerkleTree::new(test_ints.clone());

        let mut repeats = vec![23; 4];

        test_ints.append(&mut repeats);

        let mut current_layer: Vec<HashValue> = test_ints.iter().map(|x| hash(x)).collect();
        
        merkle_tree.add_element(23);

        for i in 0..merkle_tree.layers.len() {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));
            }
            current_layer = next_layer;
        }
        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 5);
    }

    #[test]
    fn add_element_works_for_trees_with_repeat_elements() {
        let mut test_ints = vec![4,8,15];
        let mut merkle_tree = MerkleTree::new(test_ints.clone());

        let mut repeats = vec![16];

        test_ints.append(&mut repeats);

        let mut current_layer: Vec<HashValue> = test_ints.iter().map(|x| hash(x)).collect();
        
        merkle_tree.add_element(16);

        for i in 0..merkle_tree.layers.len() {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));
            }
            current_layer = next_layer;
        }
        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 4);
    }

    #[test]
    fn can_generate_proof_for_element_in_even_index_in_leaves_array() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let proof = merkle_tree.generate_proof(15).unwrap();
        let mut current_hash = hash(&15);

        for i in 0..merkle_tree.layers.len() {
            current_hash = hash(&(current_hash as u128 + proof[i] as u128));
        }
        assert_eq!(merkle_tree.root, current_hash);
    }

    #[test]
    fn can_generate_proof_for_element_in_odd_index_in_leaves_array() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let proof = merkle_tree.generate_proof(16).unwrap();
        let mut current_hash = hash(&16);

        for i in 0..merkle_tree.layers.len() {
            current_hash = hash(&(current_hash as u128 + proof[i] as u128));
        }
        assert_eq!(merkle_tree.root, current_hash);
    }

    #[test]
    fn can_generate_proof_for_element_in_tree_with_repeats() {
        let test_ints = vec![4,8,15];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let proof = merkle_tree.generate_proof(15).unwrap();
        let mut current_hash = hash(&15);

        for i in 0..merkle_tree.layers.len() {
            current_hash = hash(&(current_hash as u128 + proof[i] as u128));
        }

        assert_eq!(merkle_tree.root, current_hash);
    }

    #[test]
    fn cant_generate_proof_for_element_not_in_tree() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let proof = merkle_tree.generate_proof(23);

        assert_eq!(proof, None);
    }

    #[test]
    fn proof_validation_works_for_valid_proof() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let elem = 15;

        let mut proof: Vec<HashValue> = Vec::new();

        let mut current_proof_elem_index = 2;

        for layer in merkle_tree.layers.iter() {
            if current_proof_elem_index % 2 == 0 {
                proof.push(layer[current_proof_elem_index + 1]);
            } else { proof.push(layer[current_proof_elem_index - 1]); }
            current_proof_elem_index /= 2;
        }
        
        assert!(merkle_tree.validate_proof(proof, elem))
    }

    #[test]
    fn proof_validation_doesnt_work_for_invalid_proof() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let elem = 23;

        let mut proof: Vec<HashValue> = Vec::new();

        let mut current_proof_elem_index = 2;

        for layer in merkle_tree.layers.iter() {
            if current_proof_elem_index % 2 == 0 {
                proof.push(layer[current_proof_elem_index + 1]);
            } else { proof.push(layer[current_proof_elem_index - 1]); }
            current_proof_elem_index /= 2;
        }
        
        assert!(!merkle_tree.validate_proof(proof, elem))
    }

    #[test]
    fn proof_validation_works_for_proof_generated_by_tree() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let elem = 15;

        let proof = merkle_tree.generate_proof(elem).unwrap();

        assert!(merkle_tree.validate_proof(proof, elem))
    }
}




