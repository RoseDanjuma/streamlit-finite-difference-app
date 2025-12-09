import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit App Title and Description

st.title("Numerical Derivative Visualiser")
st.write("""This interactive tool helps you understand  **forward**, **backward**, and **central** difference methods for numerical differentiation. Change the function, the point of differentiation, and the step size to see how these methods approximate the derivative.""")

# Sidebar for user inputs
st.sidebar.header("Settings")

func_choice = st.sidebar.selectbox("Choose a function f(x):", ["sin(x)"], "cos(x)" "exp(x)", "x^2", "x^3"])
x0 = st.sidebar.number_input("Point x_0:", value=1.0)
h = st.sidebar.number_input("Step size h:", value=0.1, min_value=0.001, max_value=0.5, step=0.001)

# Define the selected function f(x) 
def f(x):
    if func_choice == "sin(x)":
        return np.sin(x)
    elif func_choice == "cos(x)":
        return np.cos(x)
    elif func_choice == "exp(x)":
        return np.exp(x)
    elif func_choice == "x^2":
        return x**2
    elif func_choice == "x^3":
        return x**3
    
    # True derivative (f'(x)) for reference
    def true_derivative(x):
        if func_choice == "sin(x)":
            return np.cos(x)
        elif func_choice == "cos(x)":
            return -np.sin(x)
        elif func_choice == "exp(x)":
            return np.exp(x)
        elif func_choice == "x^2":
            return 2*x
        elif func_choice == "x^3":
            return 3*x**2
        
        st.header("Choose step size (h)")
        h = st.slider("Select step size (h)", 0.001, 0.5, 0.1, 0.001)
        st.write("You selected h =", h)
        
        #Numerical Approximations
        forward_diff = (f(x0 + h) - f(x0)) / h
        backward_diff = (f(x0) - f(x0 - h)) / h
        central_diff = (f(x0 + h) - f(x0 - h)) / (2 * h)

        true_derivative_value = true_derivative(x0)

        st.subheader("Numerical Derivative Approximations at x = {:.2f}".format(x0))
        st.write("Forward Difference: {:.6f}".format(forward_diff))
        st.write("Backward Difference: {:.6f}".format(backward_diff))
        st.write("Central Difference: {:.6f}".format(central_diff))
        st.write("True Derivative: {:.6f}".format(true_derivative_value))

        #ERROR COMPUTATION
        #Compute the absolute error between the numerical approximations and the true derivative
        
        error = np.abs(true_derivative_value - central_diff)

        #Compute maximum error for the given step size h
        max_error = np.max(error)

        #Display max error on Streamlit app
        st.write("###Maximum Error:", max_error)

    

        st.title("Finite Difference Matrix Calculator 🎛️")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
        matrix = np.loadtxt(uploaded_file, delimiter=",")
        st.write("### Uploaded Matrix")
        st.write(matrix)

        # --- METHOD SELECTOR ---
        method = st.selectbox(
        "Select Differentiation Method",
        ["Forward Difference", "Backward Difference", "Central Difference"]
        )

        h = st.number_input("Step size (h)", value=1.0)

        # --- COMPUTE DIFFERENCE ---
        def forward_diff(mat, h):
        return (mat[1:] - mat[:-1]) / h

        def backward_diff(mat, h):
        return (mat[1:] - mat[:-1]) / h  # same shape, but logically backward

        def central_diff(mat, h):
        return (mat[2:] - mat[:-2]) / (2*h)

        if st.button("Compute"):
        if method == "Forward Difference":
            result = forward_diff(matrix, h)
        elif method == "Backward Difference":
            result = backward_diff(matrix, h)
        else:
            result = central_diff(matrix, h)

        st.write("### Result")
        st.write(result)

        
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(x, true_derivative_value, label='True Derivative', color='green')
        plt.plot(x, derivative_value, label='Numerical Approximation', color='orange')
        plt.plot(x, error, label='Absolute Error', color='red'))
        plt.legend()
        st.pyplot(plt)

        xs = np.linspace(x0 - 2, x0 + 2, 400)
        ys = f(xs)

        fig, ax = plt.subplots()
        ax.plot(xs, ys, label='f(x)', color='blue')
        ax.axvline(x=x0, color='gray', linestyle='--', label='x_0')
        ax.set_title('Function and Tangent Approximations at x = {:.2f}'.format(x0))
        st.pyplot(fig)