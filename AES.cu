#include "AES.h"


// ===================== test ============================================
int main(int argc, char* argv[]) 
{
    const std::filesystem::path key_file = "/home/rohan/Parallel-AES-Algorithm-using-CUDA/key.txt";
    const std::filesystem::path text_file = "/home/rohan/Desktop/vim-test/build/random_data.txt";
    const std::filesystem::path encrypted_file = "encrypt.txt";
    const std::filesystem::path decrypted_file = "decrypt.txt";
    encrypt(key_file, text_file, encrypted_file);
    encrypt(key_file, encrypted_file, decrypted_file, false);
    return 0;
}
