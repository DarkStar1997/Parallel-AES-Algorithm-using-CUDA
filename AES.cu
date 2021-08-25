#include "AES.h"

int main(int argc, char* argv[]) {

    cudaEvent_t start, stop;
    float time_k = 0.0, time_h_to_d = 0.0, time_d_to_h = 0.0;
    std::ifstream ifs;
    Mode mode;
    
    {
        std::string mode_input = argv[1];
        if(mode_input == "encrypt")
            mode = Mode::ENCRYPTION;
        else if(mode_input == "decrypt")
            mode = Mode::DECRYPTION;
        else
        {
            std::cout << "Invalid Mode\n";
            return 0;
        }
    }
    
    ifs.open(argv[2], std::ifstream::binary);
    if(!ifs){
        std::cerr<<"Cannot open the input file"<<std::endl;
        exit(1);
    }
    ifs.seekg(0, std::ios::end);
    int infileLength = ifs.tellg();
    ifs.seekg (0, std::ios::beg);
    std::cout<<"Length of input file: "<<infileLength<<std::endl;


    int block_number = infileLength/16 ;
    int number_of_zero_pending = infileLength%16;
    aes_block* aes_block_array;

    BYTE key[16 * (14 + 1)];
    int keyLen = 0;
    int blockLen = 16;

    std::ifstream key_fp;
    key_fp.open(argv[3]);
    while(key_fp.peek()!=EOF)
    {
        key_fp>>key[keyLen];
        if(key_fp.eof())
            break;
        keyLen++;
    }

    std::cout<<keyLen<<std::endl;
    switch (keyLen)
    {
        case 16:break;
        case 24:break;
        case 32:break;
        default:printf("ERROR : keyLen should be 128, 192, 256bits\n"); return 0;
    }

    int expandKeyLen = AES_ExpandKey(key, keyLen);

    if(number_of_zero_pending != 0)
    {
        cudaError_t status = cudaMallocHost((void**)&aes_block_array, (block_number + 1) * sizeof(aes_block));
    }
    else
    {
        cudaError_t status = cudaMallocHost((void**)&aes_block_array, (block_number) * sizeof(aes_block));
    }
    char temp[16];
    FILE* out_fp;
    
    out_fp = fopen(argv[4], "wb");
    for(int i=0; i<block_number; i++){

        ifs.read(temp, 16);
        for(int j=0; j<16; j++){
            aes_block_array[i].block[j] = (unsigned char)temp[j];
        }
    }

    if(number_of_zero_pending != 0)
    {
        ifs.read(temp, number_of_zero_pending);
        for(int j=0; j<16; j++){
            aes_block_array[block_number].block[j] = (unsigned char)temp[j];
        }
        for(int j=1; j<=16-number_of_zero_pending; j++)
            aes_block_array[block_number].block[16-j] = '\0';
        block_number++;
    }


    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device name: " << prop.name << '\n';
    
    int num_sm = prop.multiProcessorCount; 

    aes_block *cuda_aes_block_array;
    BYTE *cuda_key;//, *cuda_Sbox;


    int thrdperblock = block_number/num_sm;
    if(block_number%num_sm>0)
        thrdperblock++;

    if(thrdperblock>1024){
        thrdperblock = 1024;
        num_sm = block_number/1024;
        if(block_number%1024>0){
            num_sm++;
        }
    }
    
    dim3 ThreadperBlock(thrdperblock);

    printf("num of sms: %d\nThreads per block: %d\n", num_sm, thrdperblock);
    dim3 BlockperGrid(num_sm);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    cudaMalloc(&cuda_aes_block_array, block_number*sizeof(class aes_block));
    cudaMalloc(&cuda_key,16*15*sizeof(BYTE) );
    cudaMemcpy(cuda_aes_block_array, aes_block_array, block_number*sizeof(class aes_block), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_key, key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_h_to_d, start, stop);
    
    
    if(mode == Mode::ENCRYPTION)
    {
        cudaEventRecord(start, 0);
        AES_Encrypt <<< BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key, expandKeyLen, block_number);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_k, start, stop);

        cudaEventRecord(start, 0);
        cudaMemcpy(aes_block_array, cuda_aes_block_array, block_number*sizeof(class aes_block), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_d_to_h, start, stop);

        for(int i=0; i<block_number-1; i++){
            f1printBytes(aes_block_array[i].block, blockLen, out_fp);
        }
        if(number_of_zero_pending == 0)
            f1printBytes(aes_block_array[block_number-1].block, blockLen, out_fp);
        else 
            f1printBytes(aes_block_array[block_number-1].block, blockLen, out_fp);
    }

    else if(mode == Mode::DECRYPTION)
    {
        cudaEventRecord(start, 0);
        AES_Decrypt <<< BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array, cuda_key, expandKeyLen, block_number);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_k, start, stop);

        cudaEventRecord(start, 0);
        cudaMemcpy(aes_block_array, cuda_aes_block_array, block_number*sizeof(class aes_block), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_d_to_h, start, stop);

        for(int i=0; i<block_number-1; i++){
            f3printBytes(aes_block_array[i].block, blockLen, out_fp);
        }
        if(number_of_zero_pending == 0)
            f3printBytes(aes_block_array[block_number-1].block, blockLen, out_fp);
        else 
            f3printBytes(aes_block_array[block_number-1].block, blockLen, out_fp);
    }

    cudaFree(cuda_aes_block_array);
    cudaFree(cuda_key);
    cudaFreeHost(aes_block_array);

    fclose(out_fp);

    std::cout << "Time for memory transfer from host to device = " << time_h_to_d << " ms\n";
    
    if(mode == Mode::ENCRYPTION)
    {
        std::cout << "Execution time for encryption kernel = " << time_k << " ms\n";
    }
    else if(mode == Mode::DECRYPTION)
    {
        std::cout << "Execution time for decryption kernel = " << time_k << " ms\n";
    }

    std::cout << "Time for memory transfer from device to host = " << time_d_to_h << " ms\n";
    
    return 0;
}
