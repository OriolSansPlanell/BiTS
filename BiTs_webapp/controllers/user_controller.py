import streamlit as st
from models.user_model import UserModel
from views.user_view import UserView

class UserController:
    def __init__(self):
        self.model = UserModel()
        self.view = UserView()

        if "users" not in st.session_state:
            st.session_state.users = {}

    def add_user(self, user_id, name):
        if user_id and name:
            st.session_state.users[user_id] = name

    def run(self):
        self.view.show_user_section()
        user_id, name, add_user_clicked = self.view.show_add_user_form()

        if add_user_clicked:
            self.add_user(user_id, name)

        self.view.show_users(st.session_state.users)
