//Hashvalues are of type u64. The type aliasing to HashValue is there to help with code readability.
struct HashValue(u64);


//Struct definition for branch nodes of the Merkle Tree. 
//Boxes are needed for the children due to it being a recursive data structure
struct MerkleTreeBranch {
    hash: HashValue,
    leftChild: Box<MerkleTreeBranch>,
    rightChild: Box<MerkleTreeBranch>,
}

//A Merkle Tree node can either be a leaf (and only have a hash), or a branch.
enum MerkleTreeNode {
    Leaf(HashValue),
    Branch(MerkleTreeBranch),
}

fn main() {}

