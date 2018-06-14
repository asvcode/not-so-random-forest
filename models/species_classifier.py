"""CNN for classifying the species of a tree

The network architecture trains the top layers from inception v3 on tree species data. Training proceeds in two steps. First, all layers are frozen and the penultimate layer and the last Dense layer are trained. 


"""