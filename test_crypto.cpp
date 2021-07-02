#include <cryptopp/cryptlib.h>
#include <cryptopp/rijndael.h>
#include <cryptopp/modes.h>
#include <cryptopp/files.h>
#include <cryptopp/osrng.h>
#include <cryptopp/hex.h>

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iterator>

int main()
{
    using namespace CryptoPP;
    AutoSeededRandomPool prng;
    HexEncoder encoder(new FileSink(std::cout));

    SecByteBlock key(AES::DEFAULT_KEYLENGTH);

    {
        std::string path = "../key.txt";
        std::ifstream in; in.open(path);
        std::string key_str = {std::istream_iterator<unsigned char>(in), std::istream_iterator<unsigned char>()};
        std::copy(key_str.begin(), key_str.end(), key.begin());
        in.close();
    }

    SecByteBlock iv(AES::BLOCKSIZE);

    prng.GenerateBlock(iv, iv.size());

    std::string plain;
   
    {
        std::string path = "/home/rohan/Desktop/vim-test/build/random_data.txt";
        std::ifstream in; in.open(path);
        plain = {std::istream_iterator<unsigned char>(in), std::istream_iterator<unsigned char>()};
        in.close();
    } 
   
    std::cout << "File read complete\n";
    std::string cipher, recovered;

    //std::cout << "Plain: " << plain << '\n';

    CTR_Mode<AES>::Encryption e;
    e.SetKeyWithIV(key, key.size(), iv);

    auto start_enc = std::chrono::steady_clock::now();
    StringSource s(plain, true, new StreamTransformationFilter(e, new StringSink(cipher)));
    auto end_enc = std::chrono::steady_clock::now();
    std::cout << "Time taken for encryption: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - start_enc).count() << " ms\n";

    //std::cout << "Key: ";
    //for(const auto& i : key)
    //    std::cout << i;
    //std::cout << " (" << key.size() << ")" << std::endl;

    //std::cout << "iv: ";
    //encoder.Put(iv, iv.size());
    //encoder.MessageEnd();
    //std::cout << " (" << iv.size() << ")" << std::endl;

    //std::cout << "Cipher text: ";
    //encoder.Put((const byte*)&cipher[0], cipher.size());
    //encoder.MessageEnd();
    //std::cout << std::endl;

    CTR_Mode<AES>::Decryption d;
    d.SetKeyWithIV(key, key.size(), iv);
    
    auto start_dec = std::chrono::steady_clock::now();
    StringSource s1(cipher, true, new StreamTransformationFilter(d, new StringSink(recovered)));
    auto end_dec = std::chrono::steady_clock::now();
    std::cout << "Time taken for decryption: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - start_dec).count() << " ms\n";
    //std::cout << "Recovered text: " << recovered << '\n';
    
    if(recovered == plain)
        std::cout << "Legit solution\n";
    else
        std::cout << "Invalid solution\n";
}
