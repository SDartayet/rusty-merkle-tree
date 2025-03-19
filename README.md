# Rusty Merkle Tree
## A Merkle Tree Implementation in Rust

### What is a Merkle Tree?

A Merkle Tree is a data structure, very similar to a binary (or in some implementations, an n-ary) tree. Each leave in a merkle tree contains the hash to a block or partition of some data, while every inner node contains the hash of the hashes of its children.

![Illustration of the structure of a merkle tree, courtesy of https://en.wikipedia.org/wiki/Merkle_tree](https://en.wikipedia.org/wiki/Merkle_tree)

### What's a Merkle Tree used for? 

Merkle Trees are used in peer-to-peer networks to verify the integrity of data. Real life use cases span form many forms of distributed data sharing (such as IPFS), to BitTorrent, to blockchain (namely Bitcoin).

### Specifics of the project

#### Dependencies

To build the project, you'll need:

- Rust 1.85.0

#### How to run

On command line, run 
>cargo build