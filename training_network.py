'''
Name: Joseph DiMartino
Program: Training a Simple Neural Network
Date: 10.27.2025

sigmoid: The sigmoid function is the actual math that turns the weights/input into an output of 1 or 0.
This function is also S shaped meaning it's differentiable. regular threshold functions are 1 or 0 meaning they're not differentiable.

sigmoid derivative: computes the gradients. Tells you how the output changes based on the change in weight.
when the function is very confident in score (extremely close to 1 or 0) then the derivative is very low, since
the curve of the sigmoid function is more flat. When the weight is closer to .5, the learning rate is
greater and the derivative is the largest.

notX2(): The function to train a single neuron neural network. This takes in one input, X2, which can be 1 or 0, this expects
all inputs of 1 output 0 and all inputs of 0 output a 1 (a simple not gate). This originally had a weight of 1 as well
as a bias of 1. Eventually the derivative function + error changed the weight, allowing the output to become more confident
and decrease error.

X1andX2(): function to train a single neuron neural network. Takes in two inputs X1 and X2 + one bias term. This is a
logical AND gate, same as the notX2 function, this function changes weights after 1000 iterations and slowly becomes more
and more accurate in getting an output that's expected. The input of 1 and 1 is the only way to get an output of 1.

X1xorX2(): is a two layered neural network. With only one layer the output does not ever become more accurate, every
output option actually had an output accuracy of .5, showing a second neuron was needed. This is like the in-class example
of the XOR we drew on paper, which was possible with 2 neurons and 1 layer. This gets trained to output a 1 when the
inputs are either 10 or 01
'''

import numpy as np


# a differentiable threshold function
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


# Logically this is sigmoid(x) * (1 - sigmoid(x)) but out = sigmoid(x)
def sigmoid_deriv(out):
    return out * (1 - out)


# here's the not x2 logic
def notX2():
    X = np.array([
        [1, 0],  # bias always 1 (left column)
        [1, 1],  # input can be 0 or 1 from X2 (right column)
    ])

    y = np.array([[1], [0]])  # target output is 1 or 0 if input is 0 or 1
    w = np.array([[1.0], [1.0]])  # weights where w[0] is the bias weight and w[1] is the input weight
    learning_rate = 1.0

    for epoch in range(2000):  # go through 2000 passes
        total_dE_dw = np.zeros_like(w)
        total_error = 0  # reset error for each pass

        for i in range(len(X)):
            x = X[i].reshape(1, 2)  # shape (1, 2)
            target = y[i]  # scalar target value

            # ---- Forward pass ----
            net = np.dot(x, w)  # weighted input (bias + input * weight)
            out = sigmoid(net)  # neuron output between 0–1

            error = out - target
            sq_err = error ** 2
            total_error += sq_err[0][0]  # add to total error

            # ---- Derivative wrt each weight ----
            dE_dw = 2 * error * sigmoid_deriv(out) * x.T

            # accumulate gradients across both examples
            total_dE_dw += dE_dw

        w -= learning_rate * total_dE_dw  # update weights

        # print every 100th weight and error
        if epoch % 100 == 99:
            print(f"NOT X2 Epoch {epoch + 1:2d}: weights = {w.flatten()}, total_error = {total_error:.6f}")

    print("\nFinal evaluation for XOR:")
    for i in range(len(X)):
        net = np.dot(X[i], w)
        out = sigmoid(net)
        print(f"Input X2={X[i][1]} → Output={out[0]:.4f}")


def X1andX2():
    X = np.array([
        [1, 0, 0],  # bias always 1 (left column)
        [1, 0, 1],  # inputs can be 00, 01, 10, or 11
        [1, 1, 0],
        [1, 1, 1]
    ])

    y = np.array([[0], [0], [0], [1]])  # target output is 0 for all inputs but 1 and 1
    w = np.array([[1.0], [1.0], [1.0]])
    learning_rate = 1.0

    for epoch in range(2000):  # go through 2000 passes
        total_dE_dw = np.zeros_like(w)
        total_error = 0  # reset error for each pass

        for i in range(len(X)):
            x = X[i].reshape(1, 3)  # shape (1, 3)
            target = y[i]  # scalar target value

            # ---- Forward pass ----
            net = np.dot(x, w)  # weighted input (bias + input * weight)
            out = sigmoid(net)  # neuron output between 0–1

            error = out - target
            sq_err = error ** 2
            total_error += sq_err[0][0]  # add to total error

            # ---- Derivative wrt each weight ----
            dE_dw = 2 * error * sigmoid_deriv(out) * x.T

            # accumulate gradients across both examples
            total_dE_dw += dE_dw

        w -= learning_rate * total_dE_dw  # update weights

        # print every 100th weight and error
        if epoch % 100 == 99:
            print(f"X1 and X2 Epoch {epoch + 1:2d}: weights = {w.flatten()}, total_error = {total_error:.6f}")

    print("\nFinal evaluation for X1 and X2:")
    for i in range(len(X)):
        net = np.dot(X[i], w)
        out = sigmoid(net)
        print(f"Input X1={X[i][1]}, X2={X[i][2]} → Output={out[0]:.4f}")


def X1xorX2():
    X = np.array([
        [1, 0, 0],  # bias always 1 (left column)
        [1, 0, 1],  # inputs can be 00, 01, 10, or 11
        [1, 1, 0],
        [1, 1, 1]
    ])

    y = np.array([[0], [1], [1], [0]])  # target output is 1 for 01 and 10, 0 for 00 and 11
    w_h = np.random.uniform(-1, 1, (3, 2))  # hidden weight
    w_o = np.random.uniform(-1, 1, (3, 1))  # output weight
    learning_rate = 1.0

    for epoch in range(5000):
        total_error = 0
        total_dE_dw_h = np.zeros_like(w_h)
        total_dE_dw_o = np.zeros_like(w_o)

        for i in range(len(X)):
            x = X[i].reshape(1, 3)
            target = y[i]

            # ---- Forward pass ----
            net_h = np.dot(x, w_h)
            out_h = sigmoid(net_h)

            hidden_with_bias = np.hstack(([1], out_h.flatten()))
            final_input = np.dot(hidden_with_bias, w_o)
            final_out = sigmoid(final_input)

            # ---- Error ----
            error = final_out - target
            total_error += float(error ** 2)

            # ---- Backpropagation ----
            d_output = 2 * error * sigmoid_deriv(final_out)
            d_w_o = d_output * hidden_with_bias.reshape(3, 1)

            d_hidden = d_output * w_o[1:].T * sigmoid_deriv(out_h)
            d_w_h = np.dot(x.T, d_hidden)

            total_dE_dw_h += d_w_h
            total_dE_dw_o += d_w_o

        # ---- Update Weights ----
        w_h -= learning_rate * total_dE_dw_h
        w_o -= learning_rate * total_dE_dw_o

        if epoch % 500 == 499:
            print(f"X1 XOR X2 Epoch {epoch + 1:4d}: total_error = {total_error:.6f}")

    print("\nFinal evaluation for X1 XOR X2 (2-layer network):")
    for i in range(len(X)):
        net_h = np.dot(X[i], w_h)
        out_h = sigmoid(net_h)
        hidden_with_bias = np.hstack(([1], out_h.flatten()))
        final_input = np.dot(hidden_with_bias, w_o)
        final_out = sigmoid(final_input)
        print(f"Input X1={X[i][1]}, X2={X[i][2]} → Output={final_out[0]:.4f}")


def main():
    print("--- Start Not X ---")
    notX2()
    print("--- End Not X ---\n")
    print("--- Start X1 and X2 ---")
    X1andX2()
    print("--- End X1 and X2 ---\n")
    print("--- Start X1 xor X2 ---")
    X1xorX2()
    print("--- End X1 xor X2 ---")


if __name__ == "__main__":
    main()