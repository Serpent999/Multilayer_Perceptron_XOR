%Project :XOR problem solution using 1 hidden layer multilayer Perceptron
%Engineer : Nikhil.P.Lokhande
%email:nikhil.l.1@aol.com

clear all
clc
%Input initialization
N = 5;
input = round(rand(N,2)*1);
for i = 1:N
    correct_out(i) = xor(input(i,1),input(i,2));
end
%correct_out=correct_out.';
bias = [-1 -1 -1];
lr = 0.7;
rand('state',sum(100*clock));
weights_lh = -1*2.*rand(2,2);
weights_ho = -1*2.*rand(1,2);
epochs = 10000000;
%Feed Forward Stage

for i = 1:epochs
     out = zeros(N,1);
     for j = 1:N
         %Hidden layer 1
          H1 = bias(1,1)+(input(j,1)*weights_lh(1,1)+input(j,2)*weights_lh(2,1));
          H_out(1) = 1/(1+exp(-H1));                                                %Sigmoid activation function
          %Hidden layer 2
          H2 = bias(1,1)+(input(j,1)*weights_lh(1,2)+input(j,2)*weights_lh(2,2));
          H_out(2) = 1/(1+exp(-H2));                                               %Sigmoid activation function
          %Output layer
          O1 = bias(1,1)+(H_out(1)*weights_ho(1,1)+H_out(2)*weights_ho(1,2));
          out(j)= 1/(1+exp(-O1));
          %Back Propogation Stage
          
          %Output layer error calculation
          error_O1 = out(j)*(1-out(j))*(correct_out(j)-out(j));
          %Hidden layer error calculation
          error_H1 = H_out(1)*(1-H_out(1))*weights_ho(1,1)*error_O1;
          error_H2 = H_out(2)*(1-H_out(2))*weights_ho(1,2)*error_O1;
          for k = 1:2
              weights_lh(k,1) = weights_lh(k,1) + lr*input(j,1)*error_H1;
              weights_lh(k,2) = weights_lh(k,2) + lr*input(j,2)*error_H2;
          end
          weights_ho(1,1) = weights_ho(1,1) + lr*H_out(1)*error_O1;
          weights_ho(1,2) = weights_ho(1,2) + lr*H_out(2)*error_O1;
          
     end
end

plot(correct_out,out);