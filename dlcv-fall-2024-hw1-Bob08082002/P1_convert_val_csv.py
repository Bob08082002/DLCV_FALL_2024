import pandas as pd

# Function to modify the label values
def modify_label(label):
    # Convert the label into a string and append the necessary characters
    label_str = str(label)
    
    # Modify the label based on your logic (e.g., appending 'd', 'bb', 's', etc.) #以防test.csv label不是全None
    if label_str == '14':
        return 'Noned'
    elif label_str == '2':
        return 'Nonebb'
    elif label_str == '39':
        return 'Nones'
    else:
        return 'None'  # Default case if the label doesn't match the above

# Function to modify the CSV and write to a new file
def modify_csv(input_csv_path, output_csv_path):
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)
    
    # Apply the modification to the 'label' column
    df['label'] = df['label'].apply(modify_label)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Modified CSV saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    """ 
    given one csv file:
    id,filename,label
    0,0000.jpg,14
    1,0001.jpg,2
    2,abcd.png,39
    ...

    convert it into 
    id,filename,label
    0,0000.jpg,None
    1,0001.jpg,Noneb
    2,abcd.png,None
    ...
    and write into new csv file
    """

    input_csv = "/home/zhanggenqi/DLCV/HW1/dlcv-fall-2024-hw1-Bob08082002/hw1_data/p1_data/office/val.csv"  # Path to the original CSV file
    output_csv = "/home/zhanggenqi/DLCV/HW1/dlcv-fall-2024-hw1-Bob08082002/test_ouput_dir_P1/val.csv"  # Path to save the modified CSV file
    
    modify_csv(input_csv, output_csv)