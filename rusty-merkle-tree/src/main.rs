
use core::hash;
use std::{hash::{DefaultHasher, Hash, Hasher}};

//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.
//TODO: Implement Add for further declarativeness in code

type HashValue = u64;

/// Merkle tree struct definition
/// Root and leaves are distinguished for easier access and code legibility
/// intermediate_layers represents the layers between the root and the leaves
/// unique_elements represents the amount of non-repeated elements in the leaves
struct MerkleTree {
    root: HashValue,
    layers: Vec<Vec<HashValue>>,
    unique_elements: usize
}

fn hash<T: Hash> (val: &T) -> HashValue {
    //For now I use the default hasher
    //TODO: Could be made parametric
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    let hash = hasher.finish();
    hash
}

fn closestBiggerPowerOf2 (num: f64) -> u32 {
    let exp = f64::log2(num).ceil();
    f64::powf(2.0, exp) as u32


}

impl MerkleTree {
    ///Takes an index with the position of the element if the leaves of the tree were made into an array
    ///Returns an array of hashes, in order from lowest to highest layer in the tree they're in, needed to validate the element is part of the tree
    /*fn createMerkleProof(&self, elem_index: u32) -> Vec<HashValue> {
        let mut currentNode = newPointerWith(self.root.clone());
        let mut proofHash: HashValue;
        let mut currentSubTreeIndex = elem_index;
        let mut currentSubTreeLeaves = self.size;
        let mut proof:Vec<HashValue> = vec![0; f64::log2(currentSubTreeLeaves as f64) as usize];
        //let mut currentSubTreeIndex = elem_index;

        while currentSubTreeLeaves > 1 {
            if currentSubTreeIndex > currentSubTreeLeaves / 2 {
                currentSubTreeIndex -= currentSubTreeLeaves / 2;
                proofHash = currentNode.borrow().leftChild.unwrap().borrow_mut().hash;
                currentNode = currentNode.borrow().rightChild.unwrap();
            } else {
                proofHash = currentNode.borrow().rightChild.unwrap().borrow().hash;
                currentNode = currentNode.borrow().leftChild.unwrap();
            }
            currentSubTreeLeaves /= 2;
            proof[f64::log2(currentSubTreeLeaves as f64) as usize] = proofHash;
        } 

        proof
    }*/

    ///Add an element to an already existing tree
    ///Takes an element of any hashable type
    /*fn addElement<T: Hash>(&mut self, element: T) {

        let mut currentNode= &mut self.root;
        let mut noRepeats = true;

        let mut leftChild: &mut Box<MerkleTreeNode>;
        let mut rightChild: &mut Box<MerkleTreeNode>;


        //I go through the entire Merkle Tree checking for redundant nodes 
        //(those that are copies of others, simply to make the tree balanced)
        //If I find a node that's a repeat, that means only the node that's leftmost on that subtree can be the hash of an original element
        while(noRepeats && currentNode.leftChild != None && currentNode.rightChild != None) {
            leftChild = currentNode.leftChild.as_mut().unwrap();
            rightChild = currentNode.rightChild.as_mut().unwrap();
            if leftChild.hash == rightChild.hash && leftChild.leftChild == rightChild.leftChild && leftChild.rightChild == leftChild.rightChild { noRepeats = false; break; }
            currentNode = rightChild;
        }

        let elem_hash = hash(&element);

        //If there are repeats, I go to the second element of the subtree; the leftmost element guaranteed to be a copy
        /*if !noRepeats {
            while currentNode.leftChild != None {
                currentNode = currentNode.leftChild.as_mut().unwrap();
            }
            currentNode = currentNode.parent.as_mut().unwrap();            
            currentNode.hash = elem_hash;
            currentNode = (*currentNode.parent.as_mut().unwrap()).as_mut();


            while currentNode.parent != None {
                currentNode.hash = hash(&(currentNode.leftChild.clone().unwrap().hash as u128 + currentNode.rightChild.clone().unwrap().hash as u128));
                currentNode = currentNode.parent.as_mut().unwrap();
            }
            (*currentNode).hash = hash(&(currentNode.leftChild.clone().unwrap().hash as u128 + currentNode.rightChild.clone().unwrap().hash as u128));

        } */
        
        //If there are no repeats, I need to create a whole new subtree, and incorporate that to the tree
        else {


            let mut elementVec: Vec<&T> = Vec::with_capacity(self.size as usize);

            for i in 0..self.size {
                elementVec.push(&element);
            }

            let new_node = MerkleTreeNode::new(elementVec);

            let new_root_hash = hash(&(self.root.hash as u128 + new_node.hash as u128));

            let new_root = MerkleTreeNode { hash: new_root_hash, leftChild: Some(Box::new(self.root.clone())), rightChild: Some(Box::new(new_node)), parent: None };

            self.root = new_root;
            self.size *= 2;

        }
        1+1;
    }*/

    /// Merkle tree creation function
    /// Input: A vector of data
    /// Output: a Merkle tree with the hashes of said data as its leaves
    pub fn new<T: Hash> (data: Vec<T>) -> MerkleTree {

        // Hash every element
        let mut hashes: Vec<HashValue> = data.iter().map(| elem | hash(elem)).collect();

        // The amount of leaves always has to be a power of two so the tree remains balanced
        // We achieve this by copying the last element the necessary amount of times to make the size of the leaves array a power of 23
        let last_elem = *hashes.last().unwrap();
        let repetitions: u64 = closestBiggerPowerOf2(data.len() as f64) as u64 - data.len() as u64;
        let last_elem_position = hashes.len();

        if repetitions > 0 {
            let mut last_elem_repetitions: Vec<HashValue> = vec![last_elem; repetitions as usize];
            hashes.append(&mut last_elem_repetitions);
        }

        let mut tree = MerkleTree { root: hashes[0].clone(), layers: MerkleTree::createLayers(hashes), unique_elements: last_elem_position };

        if last_elem_position > 1 {
            tree.root = hash(&((*tree.layers.last().unwrap())[0] as u128 + (*tree.layers.last().unwrap())[1] as u128));
        } 
        tree
    }

    /// Function that dynamically adds an element to the merkle tree
    /// Input: an element of a hashable type
    /// Output: none, the tree is modified dybamically
    pub fn addElement<T: Hash>(&mut self, elem: T) {
        let leaves_count = self.layers[0].len();

        // I check if the amount of unique elements is already a power of 2, or if I had to fill with repeats
        // If its a number of two, I create a new subtree (as a vector of vectors) and append the elements of the new subtree to the corresponding layer vector
        // If not, I replace the first repeated instance of an element with the new element
        let repeats_present = leaves_count > self.unique_elements;

        if (!repeats_present) {
            let mut new_layers = MerkleTree::createLayers(vec![hash(&elem); leaves_count]);
            for (layer_old, layer_new) in self.layers.iter_mut().zip(new_layers.iter_mut()) {
                layer_old.append(layer_new);
            }
            let hash_pre_root_left = hash(&(self.layers.last().unwrap()[0] as u128 + self.layers.last().unwrap()[1] as u128 ));
            let hash_pre_root_right = hash(&(self.layers.last().unwrap()[2] as u128 + self.layers.last().unwrap()[3] as u128 ));

            let last_layer_before_root = vec![hash_pre_root_left, hash_pre_root_right];
            self.layers.push(last_layer_before_root.clone());

            self.root = hash(&(last_layer_before_root[0] as u128 + last_layer_before_root[1] as u128));
        } else {

        }
    }

    fn createLayers(hashes: Vec<HashValue>) -> Vec<Vec<HashValue>> {
        let mut current_layer = hashes.clone();

        let layers_amount = f64::log2(hashes.len() as f64) as usize;
        let mut layers: Vec<Vec<HashValue>> = Vec::new();

        // I go through each of the layers, starting with the leaves, and create the layer on top of it by adding the hashes of the two elements below, adding them, and hashing the sum
        // I need to put them in the intermediate layers array in reverse order, so the smallest layer is on top

        for i in (0..layers_amount) {
            layers.push(current_layer.clone());
            let mut next_layer = Vec::new();

            for mut j in 0..(current_layer.len()/2) {
                let new_hash = hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128));
                next_layer.push(new_hash);
            } 
            current_layer = next_layer.clone();
        }
        layers
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

        for i in (0..merkle_tree.layers.len()) {
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

        

        for i in (0..merkle_tree.layers.len()) {
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

        for i in (0..merkle_tree.layers.len()) {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for mut j in 0..current_layer.len()/2 {
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
        
        merkle_tree.addElement(23);

        for i in (0..merkle_tree.layers.len()) {
            for j in 0..current_layer.len() {
                assert_eq!(merkle_tree.layers[i][j], current_layer[j]);   
            }
            let mut next_layer: Vec<HashValue> = Vec::with_capacity(current_layer.len()/2);
            for mut j in 0..current_layer.len()/2 {
                next_layer.push(hash(&(current_layer[j*2] as u128 + current_layer[j*2+1] as u128)));

            }
            current_layer = next_layer;
        }
        let root_hash = current_layer[0];
        assert_eq!(merkle_tree.root, root_hash);
        assert_eq!(merkle_tree.unique_elements, 4);
    }
    /*
    #[test]
    fn proof_generation_is_valid() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints.clone());

        let proof = merkle_tree.createMerkleProof(3);

        let mut hashed_ints = Vec::new();

        for int in test_ints.iter() {
            hashed_ints.push(hash(int));
        }
        
        let hash1 = hash(&(hashed_ints[0] as u128 + hashed_ints[1] as u128));
        let hash2 = hash(&(hashed_ints[2] as u128 + hashed_ints[3] as u128));
    
        let hash_root = hash(&(hash1 as u128 + hash2 as u128));
         
        assert_eq!(proof[0], hashed_ints[3]);
        assert_eq!(proof[1], hash1);
        
    }
    
    #[test]
    fn merkle_tree_generation_works_for_uneven_amount_of_ints() {
        let test_ints = vec![4,8,15,16,23];
        let merkle_tree = MerkleTree::new(test_ints);

        let test_ints = vec![4,8,15,16,23];
        let mut hashed_ints = Vec::new();

        for int in test_ints.iter() {
            hashed_ints.push(hash(int));
        }
        
        let hash1 = hash(&(hashed_ints[0] as u128 + hashed_ints[1] as u128));
        let hash2 = hash(&(hashed_ints[2] as u128 + hashed_ints[3] as u128));
        let hash3 = hash(&(hashed_ints[4] as u128 + hashed_ints[4] as u128));

        let hash4 = hash(&(hash1 as u128 + hash2 as u128));
        let hash5 = hash(&(hash3 as u128 + hash3 as u128));
    
        let hash_root = hash(&(hash4 as u128 + hash5 as u128));
        
        
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().leftChild.unwrap().leftChild.unwrap().hash, hashed_ints[0]);
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().leftChild.unwrap().rightChild.unwrap().hash, hashed_ints[1]);
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().rightChild.unwrap().leftChild.unwrap().hash, hashed_ints[2]);
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().rightChild.unwrap().rightChild.unwrap().hash, hashed_ints[3]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().leftChild.unwrap().leftChild.unwrap().hash, hashed_ints[4]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().leftChild.unwrap().rightChild.unwrap().hash, hashed_ints[4]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().rightChild.unwrap().leftChild.unwrap().hash, hashed_ints[4]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().rightChild.unwrap().rightChild.unwrap().hash, hashed_ints[4]);



        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().leftChild.unwrap().hash, hash1);
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().rightChild.unwrap().hash, hash2);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().leftChild.unwrap().hash, hash3);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().rightChild.unwrap().hash, hash3);

        
        assert_eq!(merkle_tree.root.leftChild.unwrap().hash, hash4);
        assert_eq!(merkle_tree.root.rightChild.unwrap().hash, hash5);


        assert_eq!(merkle_tree.root.hash, hash_root);

    }

    //Test tree creation on a size 2 vector
    

    //Test tree creation on a size 4 vector
   

    #[test]
    fn full_merkle_tree_can_add_element() {
        let test_ints = vec![4,8];
        let mut tree = MerkleTree::new(test_ints.clone());

        tree.addElement(15);

        let hash1 = hash(&(hash(&test_ints[0]) as u128 + hash(&test_ints[1]) as u128));
        let hash2 = hash(&(hash(&15) as u128 + hash(&15) as u128));

        let root_hash = hash(&(hash1 as u128 + hash2 as u128));


        assert_eq!(tree.root.leftChild.clone().unwrap().leftChild.unwrap().hash, hash(&test_ints[0]));
        assert_eq!(tree.root.leftChild.clone().unwrap().rightChild.unwrap().hash, hash(&test_ints[1]));
        assert_eq!(tree.root.rightChild.clone().unwrap().leftChild.unwrap().hash, hash(&15));
        assert_eq!(tree.root.rightChild.clone().unwrap().rightChild.unwrap().hash, hash(&15));

        assert_eq!(tree.root.leftChild.unwrap().hash, hash1);
        assert_eq!(tree.root.rightChild.unwrap().hash, hash2);

        assert_eq!(tree.root.hash, root_hash);
    }

    #[test]
    fn non_full_merkle_tree_adds_element_in_redundant_position() {
        let test_ints = vec![4,8,15];
        let mut tree = MerkleTree::new(test_ints.clone());

        tree.addElement(16);

        let hash1 = hash(&(hash(&test_ints[0]) as u128 + hash(&test_ints[1]) as u128));
        let hash2 = hash(&(hash(&test_ints[2]) as u128 + hash(&16) as u128));

        let root_hash = hash(&(hash1 as u128 + hash2 as u128));


        assert_eq!(tree.root.leftChild.clone().unwrap().leftChild.unwrap().hash, hash(&test_ints[0]));
        assert_eq!(tree.root.leftChild.clone().unwrap().rightChild.unwrap().hash, hash(&test_ints[1]));
        assert_eq!(tree.root.rightChild.clone().unwrap().leftChild.unwrap().hash, hash(&test_ints[2]));
        assert_eq!(tree.root.rightChild.clone().unwrap().rightChild.unwrap().hash, hash(&16));

        assert_eq!(tree.root.leftChild.unwrap().hash, hash1);
        assert_eq!(tree.root.rightChild.unwrap().hash, hash2);

        assert_eq!(tree.root.hash, root_hash);
    }*/

}




