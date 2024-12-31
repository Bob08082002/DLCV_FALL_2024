import pandas as pd

# Function to calculate accuracy
def calculate_accuracy(gt_csv, pred_csv):
    # Read the ground truth and prediction CSV files
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    # Merge the two DataFrames on 'id' and 'filename' to ensure alignment
    merged_df = pd.merge(gt_df, pred_df, on=['id', 'filename'], suffixes=('_gt', '_pred'))

    # Compare ground truth and predicted labels
    correct_predictions = (merged_df['label_gt'] == merged_df['label_pred']).sum()

    # Calculate accuracy
    accuracy = correct_predictions / len(merged_df) * 100

    return accuracy

# Example usage
if __name__ == "__main__":
    '''
    given GT and pred csv file, evaluate accuracy. Both are in format below:
    id,filename,label
    0,0000.jpg,14
    1,0001.jpg,2
    2,abcd.png,39
    ...
    '''
    gt_csv_path = "/home/zhanggenqi/DLCV/HW1/dlcv-fall-2024-hw1-Bob08082002/hw1_data/p1_data/office/val.csv"      # Path to the GT.csv file
    pred_csv_path = "/home/zhanggenqi/DLCV/HW1/dlcv-fall-2024-hw1-Bob08082002/test_ouput_dir_P1/val_pred.csv"  # Path to the pred.csv file

    accuracy = calculate_accuracy(gt_csv_path, pred_csv_path)
    print(f"Accuracy: {accuracy:.2f}%")