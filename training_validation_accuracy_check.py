import pickle

# Load training history
try:
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Extract last accuracy values
    train_acc = history.get('accuracy', [None])[-1]
    val_acc = history.get('val_accuracy', [None])[-1]

    if train_acc is not None and val_acc is not None:
        print(f"✅ Final Training Accuracy: {train_acc * 100:.2f}%")
        print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
    else:
        print("❌ Accuracy data missing in 'history.pkl'.")

    print("ℹ️ Test accuracy not found in history. Check your evaluation script separately.")

except FileNotFoundError:
    print("❌ Error: 'history.pkl' file not found. Train the model first!")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
