
use core::hash;
use std::hash::{DefaultHasher, Hash, Hasher};

//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.
//TODO: Implement Add for further declarativeness in code

#[derive(Clone)]
struct HashValue(u64);


//Struct definition for branch nodes of the Merkle Tree. 
#[derive(Clone)]
struct MerkleTreeNode{
    hash: HashValue,
    leftChild: Option<Box<MerkleTreeNode>>,
    rightChild: Option<Box<MerkleTreeNode>>,
}

//Takes a vector with data, and returns a merkle tree derived from the hashes of each element in the vector
fn createMerkleTreeFromData<T: Hash>(data: Vec<T>) -> MerkleTreeNode {
    
    
    let mut current_layer = Vec::new();

    //Create a vector with the leaves of the tree to be made, by hashing each data block and pushing it onto the vector as a leaf
    for block in data.iter() {
        //Create a hasher to do the hashing
        //For now I use the default hasher; could be made parametric to choose a hashing function
        let mut hasher = DefaultHasher::new();
        block.hash(&mut hasher);
        current_layer.push(MerkleTreeNode{ hash: HashValue(hasher.finish()), leftChild: None, rightChild: None } );
    }

    
    //Current layer is the current layer of nodes I'm processing. Next layer is the one I'm building based on those blocks.
    //When there's only one left, that one is the root, and I'm done
    //TODO: Currently only works if the vector size is a factor of two. Research further on how to appropriately handle cases where it isn't
    while current_layer.len() > 1 {

        //If number of elements is odd, I clone the last element and add it as another block to hash
        if current_layer.len() % 2 != 0 { current_layer.push(current_layer.last().unwrap().clone()); }

        let mut next_layer = Vec::new();

        let mut current_layer_iter = current_layer.iter();
        while let Some(node_left) = current_layer_iter.next() {

            //Each node has two child nodes, so I get the next node as well
            let node_right = current_layer_iter.next().unwrap();

            //Calculate the sum of the hashes
            let mut hasher = DefaultHasher::new();

            //Values need to be casted to u128 to prevent overflow
            (node_left.hash.0 as u128 + node_right.hash.0 as u128).hash(&mut hasher);

            let hash_sum = hasher.finish();

            //Create the new node and push it onto the next layer
            let new_node = MerkleTreeNode { hash: HashValue(hash_sum), leftChild: Some(Box::new(node_left.clone())), rightChild: Some(Box::new(node_right.clone())) };
            next_layer.push(new_node); 
        }

        //Swap the current layer to be the layar we just created, and next layer to be a new layer
        current_layer = next_layer;
    }

    //Because of how the while is structured, only the root will be left in the vector.
    //I get the first element and return it
    let root = current_layer.first().unwrap().clone();
    root

    
    


}
//Methods that make more sense to be methods of the struct itself
impl MerkleTreeNode {
    //Function that checks if a hash is in the merkle tree
    fn containsHash(&self, hash: u64) -> bool {
        //Pattern match on (None, None) because pattern matching on boxes is currently experimental
        //If both children are None, the node is a leaf, so I return whether it contains the hash
        if let (None, None) = (self.leftChild.clone(), self.rightChild.clone()) {
            self.hash.0 == hash
        } else { 
            //If node is branch, I have to check its children too
            self.hash.0 == hash || self.leftChild.clone().unwrap().containsHash(hash) || self.rightChild.clone().unwrap().containsHash(hash) }
        
        
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
        let merkle_tree_42 = createMerkleTreeFromData(vec![test_int]);
        let mut hasher = DefaultHasher::new();
        test_int.hash(&mut hasher);
        assert_eq!(merkle_tree_42.hash.0, hasher.finish());

    }

    //Test tree creation on a size 2 vector
    #[test]
    fn merkle_tree_generation_works_for_two_ints() {
        let test_ints = vec![4,2];
        let merkle_tree_42 = createMerkleTreeFromData(test_ints);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        4.hash(&mut hasher1);
        2.hash(&mut hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();

        let mut hasher_root = DefaultHasher::new();
        (hash1 as u128 + hash2 as u128).hash(&mut hasher_root);

        let hash_root = hasher_root.finish();


        let left_child_hash = merkle_tree_42.leftChild.unwrap().hash.0;
        assert_eq!(left_child_hash, hash1);

        let right_child_hash = merkle_tree_42.rightChild.unwrap().hash.0;
        assert_eq!(right_child_hash, hash2);

        assert_eq!(merkle_tree_42.hash.0, hash_root);

    }

    //Test tree creation on a size 4 vector
    #[test]
    fn merkle_tree_generation_works_for_four_ints() {
        let test_ints = vec![4,8,15,16];
        let merkle_tree = createMerkleTreeFromData(test_ints);

        let test_ints = vec![4,8,15,16];
        let mut hashed_ints = Vec::new();

        for int in test_ints.iter() {
            let mut hasher = DefaultHasher::new();

            int.hash(&mut hasher);
            hashed_ints.push(hasher.finish());
        }


        
        let mut hasher1 = DefaultHasher::new();
        (hashed_ints[0] as u128 + hashed_ints[1] as u128).hash(&mut hasher1);
        let hash1 = hasher1.finish();
        

        
        let mut hasher2 = DefaultHasher::new();
        (hashed_ints[2] as u128 + hashed_ints[3] as u128).hash(&mut hasher2);
        let hash2 = hasher2.finish();
    

        
        let mut hasher_root = DefaultHasher::new();
        (hash1 as u128 + hash2 as u128).hash(&mut hasher_root);
        let hash_root = hasher_root.finish();
        
        
        assert_eq!(merkle_tree.leftChild.clone().unwrap().leftChild.unwrap().hash.0, hashed_ints[0]);
        assert_eq!(merkle_tree.leftChild.clone().unwrap().rightChild.unwrap().hash.0, hashed_ints[1]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().leftChild.unwrap().hash.0, hashed_ints[2]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().rightChild.unwrap().hash.0, hashed_ints[3]);
        
        assert_eq!(merkle_tree.leftChild.unwrap().hash.0, hash1);
        assert_eq!(merkle_tree.rightChild.unwrap().hash.0, hash2);


        assert_eq!(merkle_tree.hash.0, hash_root);

    }

    #[test]
    fn merkle_tree_generation_works_for_uneven_amount_of_ints() {
        let test_ints = vec![4,8,15,16,23];
        let merkle_tree = createMerkleTreeFromData(test_ints);

        let test_ints = vec![4,8,15,16,23];
        let mut hashed_ints = Vec::new();

        for int in test_ints.iter() {
            let mut hasher = DefaultHasher::new();

            int.hash(&mut hasher);
            hashed_ints.push(hasher.finish());
        }


        
        let mut hasher1 = DefaultHasher::new();
        (hashed_ints[0] as u128 + hashed_ints[1] as u128).hash(&mut hasher1);
        let hash1 = hasher1.finish();
        

        
        let mut hasher2 = DefaultHasher::new();
        (hashed_ints[2] as u128 + hashed_ints[3] as u128).hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let mut hasher3 = DefaultHasher::new();
        (hashed_ints[4] as u128 + hashed_ints[4] as u128).hash(&mut hasher3);
        let hash3 = hasher3.finish();

        let mut hasher4 = DefaultHasher::new();
        (hash1 as u128 + hash2 as u128).hash(&mut hasher4);
        let hash4 = hasher4.finish();

        let mut hasher5 = DefaultHasher::new();
        (hash3 as u128 + hash3 as u128).hash(&mut hasher5);
        let hash5 = hasher5.finish();
    

        
        let mut hasher_root = DefaultHasher::new();
        (hash4 as u128 + hash5 as u128).hash(&mut hasher_root);
        let hash_root = hasher_root.finish();
        
        
        assert_eq!(merkle_tree.leftChild.clone().unwrap().leftChild.unwrap().leftChild.unwrap().hash.0, hashed_ints[0]);
        assert_eq!(merkle_tree.leftChild.clone().unwrap().leftChild.unwrap().rightChild.unwrap().hash.0, hashed_ints[1]);
        assert_eq!(merkle_tree.leftChild.clone().unwrap().rightChild.unwrap().leftChild.unwrap().hash.0, hashed_ints[2]);
        assert_eq!(merkle_tree.leftChild.clone().unwrap().rightChild.unwrap().rightChild.unwrap().hash.0, hashed_ints[3]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().leftChild.unwrap().leftChild.unwrap().hash.0, hashed_ints[4]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().leftChild.unwrap().rightChild.unwrap().hash.0, hashed_ints[4]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().rightChild.unwrap().leftChild.unwrap().hash.0, hashed_ints[4]);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().rightChild.unwrap().rightChild.unwrap().hash.0, hashed_ints[4]);



        assert_eq!(merkle_tree.leftChild.clone().unwrap().leftChild.unwrap().hash.0, hash1);
        assert_eq!(merkle_tree.leftChild.clone().unwrap().rightChild.unwrap().hash.0, hash2);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().leftChild.unwrap().hash.0, hash3);
        assert_eq!(merkle_tree.rightChild.clone().unwrap().rightChild.unwrap().hash.0, hash3);

        
        assert_eq!(merkle_tree.leftChild.unwrap().hash.0, hash4);
        assert_eq!(merkle_tree.rightChild.unwrap().hash.0, hash5);


        assert_eq!(merkle_tree.hash.0, hash_root);

    }

    #[test]
    fn merkle_tree_contains_one_hash_in_root() {
        let test_int = 42;
        let merkle_tree_42 = createMerkleTreeFromData(vec![test_int]);
        let mut hasher = DefaultHasher::new();
        test_int.hash(&mut hasher);
        assert!(merkle_tree_42.containsHash(hasher.finish()));
    }


    #[test]
    fn merkle_tree_contains_hashes_in_leaves() {
        let test_ints = vec![4,2];
        let merkle_tree_42 = createMerkleTreeFromData(test_ints);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        4.hash(&mut hasher1);
        2.hash(&mut hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();

        assert!(merkle_tree_42.containsHash(hash1));
        assert!(merkle_tree_42.containsHash(hash2));

    }
}




