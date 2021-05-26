#include "AES.h"
#include <chrono>

static int flag = 0;
void f2printBytes(BYTE b[], int len, FILE* fp) {
    int i;
    for (i = 0; i < len; i++) {
        fprintf(fp, "%c", b[i]);
        if (b[i] == '\n') flag++;
    }
    //    cout << hex << b[i] << " " ;
    // fprintf(fp, "\n");
}
void f3printBytes(BYTE b[], int len, FILE* fp) {
    int i;
    for (i = 0; i < len; i++) {
        if (b[i] == '\0') {
            return;
        }
        fprintf(fp, "%c", b[i]);
        // printf("%x ", b[i]);
        if (b[i] == '\n') flag++;
    }
    //    cout << hex << b[i] << " " ;
    // fprintf(fp, "\n");
}

// ===================== test ============================================
int main(int argc, char* argv[]) {

    std::ifstream ifs;
    ifs.open(argv[1], std::ifstream::binary);
    if (!ifs) {
        std::cerr << "Cannot open the input file" << std::endl;
        exit(1);
    }
    ifs.seekg(0, std::ios::end);
    int infileLength = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::cout << "Length of input file: " << infileLength << std::endl;


    int block_number = infileLength / 16;
    int number_of_zero_pending = infileLength % 16;
    std::vector<aes_block> aes_block_array;

    std::vector<BYTE> key(16 * 15);
    thrust::device_vector<BYTE> cuda_key; cuda_key.reserve(key.size());
    int keyLen = 0;
    int blockLen = 16;

    std::ifstream key_fp;
    key_fp.open(argv[2]);
    while (key_fp.peek() != EOF) {
        key_fp >> key[keyLen];
        if (key_fp.eof()) break;
        keyLen++;
    }

    std::cout << keyLen << std::endl;
    switch (keyLen) {
        case 16:
            break;
        case 24:
            break;
        case 32:
            break;
        default:
            printf("ERROR : keyLen should be 128, 192, 256bits\n");
            return 0;
    }

    int expandKeyLen = AES_ExpandKey(key, keyLen);

    if (number_of_zero_pending != 0)
        aes_block_array = std::vector<aes_block>(block_number + 1);
    else
        aes_block_array = std::vector<aes_block>(block_number);
    thrust::device_vector<aes_block> cuda_aes_block_array;
    cuda_aes_block_array.reserve(aes_block_array.size());
    char temp[16];
    FILE* en_fp;
    FILE* de_fp;
    en_fp = fopen("encrypt.txt", "wb");
    de_fp = fopen("decrypt.txt", "wb");
    for (int i = 0; i < block_number; i++) {
        ifs.read(temp, 16);
        for (int j = 0; j < 16; j++) {
            aes_block_array[i].block[j] = (unsigned char)temp[j];
        }
    }
    if (number_of_zero_pending != 0) {
        ifs.read(temp, number_of_zero_pending);
        for (int j = 0; j < 16; j++) {
            aes_block_array[block_number].block[j] = (unsigned char)temp[j];
        }
        for (int j = 1; j <= 16 - number_of_zero_pending; j++)
            aes_block_array[block_number].block[16 - j] = '\0';
        block_number++;
    }

    cudaSetDevice(0); 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sm = prop.multiProcessorCount;


    int thrdperblock = block_number / num_sm;
    if (block_number % num_sm > 0) thrdperblock++;

    if (thrdperblock > 1024) {
        thrdperblock = 1024;
        num_sm = block_number / 1024;
        if (block_number % 1024 > 0) {
            num_sm++;
        }
    }
    dim3 ThreadperBlock(thrdperblock);

    printf("num of sms: %d\nThreads per block: %d\n", num_sm, thrdperblock);
    dim3 BlockperGrid(num_sm);
    auto start_enc = std::chrono::steady_clock::now();
    thrust::copy(aes_block_array.begin(), aes_block_array.end(), cuda_aes_block_array.begin());
    thrust::copy(key.begin(), key.end(), cuda_key.begin());
    
    AES_Encrypt<<<BlockperGrid, ThreadperBlock>>>(
        thrust::raw_pointer_cast(cuda_aes_block_array.data()), thrust::raw_pointer_cast(cuda_key.data()), expandKeyLen, block_number);
    auto end_enc = std::chrono::steady_clock::now();
    std::cout << "Time taken for encryption is " << std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - start_enc).count() << " ms\n";

    thrust::copy(cuda_aes_block_array.begin(), cuda_aes_block_array.end(), aes_block_array.begin());

    //std::cout << "Writing encrypted output to file\n";
    //for (int i = 0; i < block_number - 1; i++) {
    //    f1printBytes(aes_block_array[i].block, blockLen, en_fp);
    //}
    //if (number_of_zero_pending == 0)
    //    f1printBytes(aes_block_array[block_number - 1].block, blockLen, en_fp);
    //else
    //    f1printBytes(aes_block_array[block_number - 1].block, blockLen, en_fp);
    //std::cout << "Writing finished\n";

    auto start_dec = std::chrono::steady_clock::now();
    AES_Decrypt<<<BlockperGrid, ThreadperBlock>>>(
        thrust::raw_pointer_cast(cuda_aes_block_array.data()), thrust::raw_pointer_cast(cuda_key.data()), expandKeyLen, block_number);

    thrust::copy(cuda_aes_block_array.begin(), cuda_aes_block_array.end(), aes_block_array.begin());
    
    auto end_dec = std::chrono::steady_clock::now();
    std::cout << "Time taken for decryption is " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - start_dec).count() << " ms\n";

    std::cout << "Writing decrypted output to file\n";
    for (int i = 0; i < block_number - 1; i++) {
        f2printBytes(aes_block_array[i].block, blockLen, de_fp);
    }
    if (number_of_zero_pending == 0)
        f2printBytes(aes_block_array[block_number - 1].block, blockLen, de_fp);
    else
        f3printBytes(aes_block_array[block_number - 1].block, blockLen, de_fp);
    std::cout << "Writing finished\n";

    AES_Done();
    fclose(en_fp);
    fclose(de_fp);


    return 0;
}
