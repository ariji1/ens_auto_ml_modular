import streamlit as st
import pandas as pd
import io
import chardet
from io import BytesIO
import gzip



def main():
    file_bytes = st.file_uploader("Upload a file",encoding=None)
    data_load_state = st.text("Upload your data")
    # file_bytes = st.file_uploader("Upload a file")
    if file_bytes is not None:
        result = chardet.detect(file_bytes.read())
        st.write(result['encoding'])
        file_bytes.seek(0)
        main_data = file_bytes.read()
        # with BytesIO() as myio:

        st.write(pd.read_csv(BytesIO(main_data),sep=",",encoding=result['encoding']))


if __name__ == '__main__':
    main()