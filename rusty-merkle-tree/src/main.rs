
use core::hash;
use std::hash::{DefaultHasher, Hash, Hasher};

//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.
//TODO: Implement Add for further declarativeness in code

type HashValue = u64;


//Struct definition for branch nodes of the Merkle Tree. 
#[derive(Clone, PartialEq)]
struct MerkleTreeNode{
    hash: HashValue,
    leftChild: Option<Box<MerkleTreeNode>>,
    rightChild: Option<Box<MerkleTreeNode>>,
}

struct MerkleTree {
    root: MerkleTreeNode,
    size: u32,
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

impl MerkleTreeNode {
    ///Takes a vector of data of any type that can be hashed
    ///Returns a subtree of a Merkle tree that contains the has of one element in its leaves, in order
    fn new<T: Hash> (data: Vec<T>) -> MerkleTreeNode {
        let mut current_layer = Vec::new();

        //Create a vector with the leaves of the tree to be made, by hashing each data block and pushing it onto the vector as a leaf
        for block in data.iter() {
            current_layer.push(MerkleTreeNode{ hash: hash(block), leftChild: None, rightChild: None } );
        }


        //Current layer is the current layer of nodes I'm processing. Next layer is the one I'm building based on those blocks.
        //When there's only one left, that one is the root, and I'm done
        //TODO: Currently only works if the vector size is a factor of two. Research further on how to appropriately handle cases where it isn't
        while current_layer.len() > 1 {

            //If number of elements is odd, I clone the last element and add it as another block to hash
            if current_layer.len() % 2 != 0 { 

                let last_element = current_layer.last().unwrap();
                let last_element_clone = MerkleTreeNode {
                    hash: last_element.hash,
                    leftChild: last_element.leftChild.clone(),
                    rightChild: last_element.rightChild.clone()
                };

                current_layer.push(last_element_clone); 
            }

            let mut next_layer = Vec::new();

            let mut current_layer_iter = current_layer.iter();
            while let Some(node_left) = current_layer_iter.next() {

                //Each node has two child nodes, so I get the next node as well
                let node_right = current_layer_iter.next().unwrap();

                //Calculate the sum of the hashes
                //Values need to be casted to u128 to prevent overflow
                let hash_sum = hash(&(node_left.hash as u128 + node_right.hash as u128));

                //Create the new node and push it onto the next layer
                let new_node = MerkleTreeNode { hash: hash_sum, leftChild: Some(Box::new(node_left.clone())), rightChild: Some(Box::new(node_right.clone())) };
                next_layer.push(new_node); 
            }

            //Swap the current layer to be the layar we just created, and next layer to be a new layer
            current_layer = next_layer;
        }

        //Because of how the while is structured, only the root will be left in the vector.
        //I get the first element and use it for the tree root
        let node = current_layer.first().unwrap().clone();
        node
    }
}

impl MerkleTree {
    ///Takes an index with the position of the element if the leaves of the tree were made into an array
    ///Returns an array of hashes, in order from lowest to highest layer in the tree they're in, needed to validate the element is part of the tree
    fn createMerkleProof(&self, elem_index: u32) -> Vec<HashValue> {
        let mut currentNode = self.root.clone();
        let mut proofHash: HashValue;
        let mut currentSubTreeIndex = elem_index;
        let mut currentSubTreeLeaves = self.size;
        let mut proof:Vec<HashValue> = vec![0; f64::log2(currentSubTreeLeaves as f64) as usize];
        //let mut currentSubTreeIndex = elem_index;

        while currentSubTreeLeaves > 1 {
            if currentSubTreeIndex > currentSubTreeLeaves / 2 {
                currentSubTreeIndex -= currentSubTreeLeaves / 2;
                proofHash = currentNode.leftChild.unwrap().hash;
                currentNode = *currentNode.rightChild.unwrap();
            } else {
                proofHash = currentNode.rightChild.unwrap().hash;
                currentNode = *currentNode.leftChild.unwrap();
            }
            currentSubTreeLeaves /= 2;
            proof[f64::log2(currentSubTreeLeaves as f64) as usize] = proofHash;
        } 

        proof
    }

    ///Add an element to an already existing tree
    ///Takes an element of any hashable type
    fn addElement<T: Hash>(&self, element: T) {

        let mut currentNode = self.root.clone();
        let mut repeats = false;

        //I go through the entire Merkle Tree checking for redundant nodes 
        //(those that are copies of others, simply to make the tree balanced)
        //If I find a node that's a repeat, that means only the node that's leftmost on that subtree can be the hash of an original element
        while(repeats) {
            let leftChild = currentNode.leftChild.unwrap();
            let rightChild = currentNode.rightChild.unwrap();
            if leftChild.hash == rightChild.hash && leftChild.leftChild == rightChild.leftChild && leftChild.rightChild == leftChild.rightChild { repeats = true; }
            currentNode = *rightChild.clone();
        }

        let elem_hash = hash(&element);

        //If there are repeats, I go to the second element of the subtree; the leftmost element guaranteed to be a copy
        if repeats {
            while currentNode.leftChild != None {
                currentNode = *currentNode.rightChild.unwrap().clone();
            }
            currentNode.rightChild.unwrap().hash = elem_hash;
        } 
        
        //If there are no repeats, I need to create a whole new subtree, and incorporate that to the tree
        else {
            let new_node = MerkleTreeNode::new(vec![element]);

            let new_root_hash = hash(&(self.root.hash as u128 + new_node.hash as u128));

            let new_root = MerkleTreeNode { hash: new_root_hash, leftChild: Some(Box::new(self.root.clone())), rightChild: Some(Box::new(new_node)) };

        }
    }

    ///Takes a vector of data of any type that can be hashed
    ///Returns a Merkle tree that contains the has of one element in each leaf, in order
    fn new<T: Hash> (data: Vec<T>) -> MerkleTree {
        let size = closestBiggerPowerOf2(data.len() as f64);
        let root = MerkleTreeNode::new(data);

        let tree = MerkleTree { root, size };
        tree
    }
}


fn main() {

}

#[cfg(test)]
mod tests {
    use std::ops::DerefMut;

    use super::*;

    //Test tree with just one data block
    #[test]
    fn merkle_tree_generation_works_for_one_int() {
        let test_int = 42;
        let merkle_tree_42 = MerkleTree::new(vec![test_int]);
        assert_eq!(merkle_tree_42.root.hash, hash(&test_int));

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
    #[test]
    fn merkle_tree_generation_works_for_two_ints() {
        let test_ints = vec![4,2];
        let merkle_tree_42 = MerkleTree::new(test_ints);

        let hash1 = hash(&4);
        let hash2 = hash(&2);

        let hash_root = hash(&(hash1 as u128 + hash2 as u128));



        let left_child_hash = merkle_tree_42.root.leftChild.unwrap().hash;
        assert_eq!(left_child_hash, hash1);

        let right_child_hash = merkle_tree_42.root.rightChild.unwrap().hash;
        assert_eq!(right_child_hash, hash2);

        assert_eq!(merkle_tree_42.root.hash, hash_root);

    }

    //Test tree creation on a size 4 vector
    #[test]
    fn merkle_tree_generation_works_for_four_ints() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = MerkleTree::new(test_ints);

        let test_ints = vec![4,8,15,16];
        let mut hashed_ints = Vec::new();

        for int in test_ints.iter() {
            hashed_ints.push(hash(int));
        }

        let hash1 = hash(&(hashed_ints[0] as u128 + hashed_ints[1] as u128));
        let hash2 = hash(&(hashed_ints[2] as u128 + hashed_ints[3] as u128));
  
        let hash_root = hash(&(hash1 as u128 + hash2 as u128));
        
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().leftChild.unwrap().hash, hashed_ints[0]);
        assert_eq!(merkle_tree.root.leftChild.clone().unwrap().rightChild.unwrap().hash, hashed_ints[1]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().leftChild.unwrap().hash, hashed_ints[2]);
        assert_eq!(merkle_tree.root.rightChild.clone().unwrap().rightChild.unwrap().hash, hashed_ints[3]);
        
        assert_eq!(merkle_tree.root.leftChild.unwrap().hash, hash1);
        assert_eq!(merkle_tree.root.rightChild.unwrap().hash, hash2);


        assert_eq!(merkle_tree.root.hash, hash_root);

    }

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
    fn full_merkle_tree_can_add_element() {

        
    }
}




