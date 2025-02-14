import streamlit as st
from models.data_model import DataModel
from views.data_view import DataView

class DataController:
    def __init__(self, file_path):
        self.model = DataModel(file_path)
        self.view = DataView()

        if "data" not in st.session_state:
            st.session_state.data = None
        if "summary" not in st.session_state:
            st.session_state.summary = None

    def load_data(self):
        if st.session_state.data is None:
            st.session_state.data = self.model.load_data()
            st.session_state.summary = self.model.get_summary(st.session_state.data)

    def run(self):
        self.view.show_title()

        if self.view.show_buttons():
            self.load_data()

        self.view.show_data(st.session_state.data)
        self.view.show_summary(st.session_state.summary)
