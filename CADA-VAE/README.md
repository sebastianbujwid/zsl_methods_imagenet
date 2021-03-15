# CADA-VAE implementation

## Differences from the [original implementation](https://github.com/edgarschnfld/CADA-VAE-PyTorch):

This implementation should be equivalent with the original implementation, except for the following factors:

**VAE reparametrization:**
- Factor of 0.5 in reparametrization of VAE
    
**VAEs: loss aggregation**
- They sum over batch samples, I average

## Other

* The code should be able to run in generalized ZSL setting, however we haven't experimented with this setting much.