# Parallel-AES-Algorithm-using-CUDA

This project demonstrates AES encryption and decryption on NVIDIA GPUs using CUDA and achieves nearly 5x performance boost compared to a decent serial CPU implementation.

## Building the project
```bash
mkdir build
cd build/
cmake ..
make
```
## Run
Usage:
```baseh
./AES [mode] [input file] [key file] [output file]
```

mode - encrypt / decrypt
input file - the file to be encrypted / decrypted
key file - the key file to be taken as input
output file - the encrypted / decrypted file as output

Example of usage:

To encrypt a file input.txt with key.txt as input to generate encrypted.txt use the command:

./AES encrypt input.txt key.txt encrypted.txt

To decrypt the file encrypted.txt with key.txt as input to generated decrypted.txt use the command:

./AES decrypt encrypted.txt key.txt decrypted.txt

The files input.txt and decrypted.txt should be identical.
