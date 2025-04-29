#pragma once 
#include "./include/k2c_tensor_include.h" 
void LSTM_316_09752336144447327(k2c_tensor* lstm_input_input, k2c_tensor* dense_1_output,float* lstm_output_array,float* lstm_kernel_array,float* lstm_recurrent_kernel_array,float* lstm_bias_array,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_kernel_array,float* dense_1_bias_array); 
void LSTM_316_09752336144447327_initialize(float** lstm_output_array 
,float** lstm_kernel_array 
,float** lstm_recurrent_kernel_array 
,float** lstm_bias_array 
,float** dense_output_array 
,float** dense_kernel_array 
,float** dense_bias_array 
,float** dense_1_kernel_array 
,float** dense_1_bias_array 
); 
void LSTM_316_09752336144447327_terminate(float* lstm_output_array,float* lstm_kernel_array,float* lstm_recurrent_kernel_array,float* lstm_bias_array,float* dense_output_array,float* dense_kernel_array,float* dense_bias_array,float* dense_1_kernel_array,float* dense_1_bias_array); 
