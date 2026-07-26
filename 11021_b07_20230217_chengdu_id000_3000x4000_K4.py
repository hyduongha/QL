11021_b07_20230217_chengdu_id000_3000x4000_K4

def main(name):
    numbers = re.findall(r'\d+', name)
    input_path = "/content/drive/MyDrive/11021_b07_20230217_chengdu_id000_3000x4000_K4/" + name
    excel_path = os.path.join("/content/drive/MyDrive/11021_b07_20230217_chengdu_id000_3000x4000_K4/log"+numbers[0]+".xlsx")  # file Excel lưu
    excel_path_Vector = os.path.join("/content/drive/MyDrive/11021_b07_20230217_chengdu_id000_3000x4000_K4/logVector"+numbers[0]+".xlsx")  # file Excel lưu
    excel_path_Eig = os.path.join("/content/drive/MyDrive/11021_b07_20230217_chengdu_id000_3000x4000_K4/logEig"+numbers[0]+".xlsx")  # file Excel lưu
    output_path = "/content/drive/MyDrive/11021_b07_20230217_chengdu_id000_3000x4000_K4/out"+numbers[0]

    if not os.path.isdir(input_path):
        print(f"❌ Thư mục {input_path} không tồn tại!")
        exit()
    os.makedirs(output_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {input_path}!")
        exit()

    log_rows = []  # mỗi phần tử: (tên file, bắt đầu, kết thúc)

    for idx, file_name in enumerate(image_files, start=1):
        start = time.perf_counter()
        k = int(re.search(r"_(\d+)\.png$", file_name).group(1))

        image_path = os.path.join(input_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path} và k={k}")

        sigma_i = 0.009
        sigma_x = 8

        save_image_name = os.path.join(output_path, f"{os.path.splitext(file_name)[0]}")
        start_vecs, end_vecs, start_V_ql, end_V_ql, end_V_qpe, end_V_iqpe, evals, E_ql, E_qpe, E_iqpe, vecs, V_ql, V_qpe, V_iqpe = normalized_cuts_eigsh(file_name, image_path, save_image_name, k, sigma_i, sigma_x)
        
        append_log_excel(excel_path, file_name, start_vecs, end_vecs, start_V_ql, end_V_ql, end_V_qpe, end_V_iqpe)
        # append_eigenvalues_simple(excel_path_Eig, file_name, evals, E_ql, E_qpe, E_iqpe)
        # append_eigenvectors_row_format(excel_path_Vector, file_name, vecs, V_ql, V_qpe, V_iqpe)
        end = time.perf_counter()
        print("Thời gian xử lý 1 ảnh ",end-start)

#####################################      Chỗ sửa đường dẫn     ######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nameImage", type=str)
    args = parser.parse_args()
    main(args.nameImage)
########################################################################################################################
