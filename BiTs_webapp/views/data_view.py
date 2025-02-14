import streamlit as st

class DataView:
    def show_title(self):
        st.title("Data Dashboard")

    def show_buttons(self):
        return st.button("Load Data")

    def show_data(self, data):
        if data is not None:
            st.dataframe(data)
        else:
            st.error("No data available.")

    def show_summary(self, summary):
        if summary is not None:
            st.subheader("Data Summary")
            st.write(summary)
