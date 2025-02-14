import streamlit as st

class UserView:
    def show_user_section(self):
        st.subheader("User Management")

    def show_add_user_form(self):
        user_id = st.text_input("User ID")
        name = st.text_input("User Name")
        return user_id, name, st.button("Add User")

    def show_users(self, users):
        if users:
            st.subheader("User List")
            st.write(users)
        else:
            st.warning("No users added yet.")
