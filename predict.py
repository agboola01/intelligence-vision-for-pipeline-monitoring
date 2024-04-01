import streamlit as st
import pickle
import numpy as np
import os

# Load the model
data = pickle.load(open('models/saved_steps1.pkl','rb'))
df = pickle.load(open('models/saved_steps2.pkl','rb'))

lgr = data["model"]
lgr2 = df["model"]
def main(title = "Leak point and distance to the link point prediction".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: blue;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    st.image("Data\pipeline.png",width=700)
    st.write("""### We need some information to predict the flow out""")
if __name__ == "__main__": 
    main()
pressure_in = st.number_input("input  pressure in value", 20.0, 100.0,20.0)
pressure_out = st.number_input("input  pressure out value", 20.0, 100.0,20.0 )
flow_in = st.number_input("input  flow in value", 0.0, 1.0,0.0)
flow_out = st.number_input("input  flow out value", 0.0, 1.0,0.0)


ok = st.button("predict leak point and distance to the link point simultaneously")
if ok:
    X = np.array([[pressure_in, pressure_out, flow_in ,flow_in ]])
    X = X.astype(float)

    leak = lgr.predict(X)
    st.subheader(f"The predicted leak point is {leak[0]:.6f}")
    dist = lgr2.predict(X)
    st.subheader(f"The predicted distance is {dist[0]:.6f}metre to the leak point")

    