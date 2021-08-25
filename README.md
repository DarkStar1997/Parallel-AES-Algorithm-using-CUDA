
# Parallel-AES-Algorithm-using-CUDA

This project demonstrates AES encryption and decryption on NVIDIA GPUs using CUDA and achieves upto 6x performance boost compared to a decent serial CPU implementation.

## Building the project
```bash
mkdir build
cd build/
cmake ..
make
```
## Run
Usage:

```bash
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

Results for AES encryption

Size of Text file | Number of Characters | AES CPU | AES GPU | Speedup
|---|---|---|---|---|
98KB  | 100K | ~0ms | 0.18ms | NA |
977KB | 1M | 2ms | 0.40ms | 5.00 |
4.8MB | 5M | 6ms | 1.15ms | 5.22 |
9.6MB | 10M | 11ms | 1.99ms | 5.53 |
48MB | 50M | 49ms | 8.18ms | 5.99 |
96MB | 100M | 97ms | 16.41ms | 5.88 |

It can be seen that a maximum speedup of 6x can be reached with the GPU implementation. But it is to be noted that most of the time is taken in memory transfer between host and device. Checkout the following screenshot:

![AES CUDA](https://imgur.com/tH7YssB.png)

This can be further verified from the nvprof output:

![nvprof](https://imgur.com/o6nuitn.png)

Please note the timers for the kernel which causes the actual encrytion to take place:

![CUDA Kernel](https://imgur.com/fOepkRH.png)

Hence the actual encryption time is very negligible compared to the memory transfer between the host and device.
