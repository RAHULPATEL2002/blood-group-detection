import pickle

try:
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Defensive checks in case keys are missing
    train_acc = history.get('accuracy', [None])[-1]
    val_acc = history.get('val_accuracy', [None])[-1]
    test_acc = history.get('test_accuracy', [None])[-1]  # in case test accuracy was saved

    if train_acc is not None:
        print(f"✅ Final Training Accuracy: {train_acc * 100:.2f}%")
    else:
        print("ℹ️ Training accuracy not found in history.")

    if val_acc is not None:
        print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
    else:
        print("ℹ️ Validation accuracy not found in history.")
    
    if test_acc is not None:
        print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")
    else:
        print("ℹ️ Test accuracy not found in history.")

except FileNotFoundError:
    print("❌ Error: 'history.pkl' file not found. Train the model first!")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
