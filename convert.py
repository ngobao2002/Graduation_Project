import streamlit as st
import numpy as np
from PIL import Image
import os
import io

def pe2hex(file_path):
    try:
        with open(file_path, 'rb') as f:
            file = bytearray(f.read())
        key = bytearray(b'\0')
        hex_bytes = []
        for count, byte in enumerate(file, 1):
            xor_byte = byte ^ key[(count - 1) % len(key)]
            hex_bytes.append(f'{xor_byte:#04x}' + ('\n' if count % 16 == 0 else ' '))
        return ''.join(hex_bytes)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

def hex2img(array):
    if array.shape[1] != 16:
        st.error("The array does not have 16 columns, indicating it may not be hexadecimal data.")
        return None

    num_pixels = array.shape[0] * 16
    side_length = int(np.sqrt(num_pixels))
    side_length = 2 ** (int(np.log2(side_length)) + 1)
    new_height = num_pixels // side_length

    if new_height * side_length < num_pixels:
        new_height += 1

    reshaped_array = array.flatten()[:new_height * side_length]
    reshaped_array = np.reshape(reshaped_array, (new_height, side_length))

    im = Image.fromarray(np.uint8(reshaped_array))
    return im

st.title("EXE to Bytes and Bytes to Image Converter")

st.header("Upload EXE File")
exe_file = st.file_uploader("Choose an EXE file", type=['exe'])

if exe_file:
    exe_path = os.path.join("C:/Users/Tam Bao/Downloads/website_deploy/tmp", exe_file.name)
    with open(exe_path, 'wb') as f:
        f.write(exe_file.getbuffer())

    st.write(f"File {exe_file.name} uploaded successfully!")

    st.header("Convert EXE to Bytes")
    if st.button("Convert to Bytes"):
        bytes_output = pe2hex(exe_path)
        if bytes_output:
            bytes_file_path = os.path.join("C:/Users/Tam Bao/Downloads/website_deploy/exe_to_bytes", exe_file.name.split('.')[0] + '.bytes')
            with open(bytes_file_path, 'w') as f:
                f.write(bytes_output)
            st.success(f"Converted to bytes and saved to {bytes_file_path}")
            st.download_button("Download Bytes File", data=bytes_output, file_name=f"{exe_file.name.split('.')[0]}.bytes")
            print(bytes_output)
    st.header("Convert to Image")
    bytes_file = bytes_output

    if bytes_file:
        bytes_data = bytes_file.read().decode('utf-8').split()
        bytes_data = [int(b, 16) for b in bytes_data if b.startswith('0x')]
        array = np.array(bytes_data).reshape(-1, 16)

        st.header("Convert Bytes to Image")
        if st.button("Convert to Image"):
            image = hex2img(array)
            if image:
                st.image(image, caption='Converted Image', use_column_width=True)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                st.download_button("Download Image", data=img_byte_arr.getvalue(), file_name="converted_image.png")
    # if st.button("Convert to Image"):
    #     image = hex2img(exe_path)
    #     if image:
    #         st.image(image, caption='Converted Image', use_column_width=True)
    #         img_byte_arr = io.BytesIO()
    #         image.save(img_byte_arr, format='PNG')
    #         st.download_button("Download Image", data=img_byte_arr.getvalue(), file_name="converted_image.png")