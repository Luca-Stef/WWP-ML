#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void LSTM_316_09752336144447327(k2c_tensor* lstm_input_input, k2c_tensor* dense_1_output,float* lstm_output_array,float* lstm_kernel_array,float* lstm_recurrent_kernel_array,float* lstm_bias_array,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_kernel_array,float* dense_1_bias_array) { 

k2c_tensor lstm_output = {lstm_output_array,1,5,{5,1,1,1,1}}; 
float lstm_fwork[40] = {0}; 
int lstm_go_backwards = 0;
int lstm_return_sequences = 0;
float lstm_state[10] = {0}; 
k2c_tensor lstm_kernel = {lstm_kernel_array,2,160,{32, 5, 1, 1, 1}}; 
k2c_tensor lstm_recurrent_kernel = {lstm_recurrent_kernel_array,2,100,{20, 5, 1, 1, 1}}; 
k2c_tensor lstm_bias = {lstm_bias_array,1,20,{20, 1, 1, 1, 1}}; 

 
k2c_tensor dense_output = {dense_output_array,1,5,{5,1,1,1,1}}; 
k2c_tensor dense_kernel = {dense_kernel_array,2,25,{5,5,1,1,1}}; 
k2c_tensor dense_bias = {dense_bias_array,1,5,{5,1,1,1,1}}; 
float dense_fwork[30] = {0}; 

 
k2c_tensor dense_1_kernel = {dense_1_kernel_array,2,5,{5,1,1,1,1}}; 
k2c_tensor dense_1_bias = {dense_1_bias_array,1,1,{1,1,1,1,1}}; 
float dense_1_fwork[10] = {0}; 

 
k2c_lstm(&lstm_output,lstm_input_input,lstm_state,&lstm_kernel, 
	&lstm_recurrent_kernel,&lstm_bias,lstm_fwork, 
	lstm_go_backwards,lstm_return_sequences, 
	k2c_sigmoid,k2c_tanh); 
k2c_tensor dropout_output; 
dropout_output.ndim = lstm_output.ndim; // copy data into output struct 
dropout_output.numel = lstm_output.numel; 
memcpy(dropout_output.shape,lstm_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
dropout_output.array = &lstm_output.array[0]; // rename for clarity 
k2c_dense(&dense_output,&dropout_output,&dense_kernel, 
	&dense_bias,k2c_relu,dense_fwork); 
k2c_dense(dense_1_output,&dense_output,&dense_1_kernel, 
	&dense_1_bias,k2c_sigmoid,dense_1_fwork); 

 } 

void LSTM_316_09752336144447327_initialize(float** lstm_output_array 
,float** lstm_kernel_array 
,float** lstm_recurrent_kernel_array 
,float** lstm_bias_array 
,float** dense_output_array 
,float** dense_kernel_array 
,float** dense_bias_array 
,float** dense_1_kernel_array 
,float** dense_1_bias_array 
) { 

*lstm_output_array = k2c_read_array("LSTM_316_09752336144447327lstm_output_array.csv",5); 
*lstm_kernel_array = k2c_read_array("LSTM_316_09752336144447327lstm_kernel_array.csv",160); 
*lstm_recurrent_kernel_array = k2c_read_array("LSTM_316_09752336144447327lstm_recurrent_kernel_array.csv",100); 
*lstm_bias_array = k2c_read_array("LSTM_316_09752336144447327lstm_bias_array.csv",20); 
*dense_output_array = k2c_read_array("LSTM_316_09752336144447327dense_output_array.csv",5); 
*dense_kernel_array = k2c_read_array("LSTM_316_09752336144447327dense_kernel_array.csv",25); 
*dense_bias_array = k2c_read_array("LSTM_316_09752336144447327dense_bias_array.csv",5); 
*dense_1_kernel_array = k2c_read_array("LSTM_316_09752336144447327dense_1_kernel_array.csv",5); 
*dense_1_bias_array = k2c_read_array("LSTM_316_09752336144447327dense_1_bias_array.csv",1); 
} 

void LSTM_316_09752336144447327_terminate(float* lstm_output_array,float* lstm_kernel_array,float* lstm_recurrent_kernel_array,float* lstm_bias_array,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_kernel_array,float* dense_1_bias_array) { 

free(lstm_output_array); 
free(lstm_kernel_array); 
free(lstm_recurrent_kernel_array); 
free(lstm_bias_array); 
free(dense_output_array); 
free(dense_kernel_array); 
free(dense_bias_array); 
free(dense_1_kernel_array); 
free(dense_1_bias_array); 
} 

