//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.

use std::hash::{DefaultHasher, Hash, Hasher};
struct HashValue(u64);


//Struct definition for branch nodes of the Merkle Tree. 
//Boxes are needed for the children due to it being a recursive data structure
struct MerkleTreeBranch{
    hash: HashValue,
    leftChild: Box<Option<MerkleTreeBranch>>,
    rightChild: Box<Option<MerkleTreeBranch>>,
}
fn main() {}




