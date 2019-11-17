import numpy as np

def log_separating_line(length_table):
    filling_chars = "-" * (length_table - 2)
    separating_line = "+{}+".format(filling_chars)
    print(separating_line)
    
def log_header_line(line, length_table, oc_char='|'):
    filling_ws = " " * (length_table - len(line) - 3)
    logged_line = "{} {}{}{}".format(oc_char, line, filling_ws, oc_char)
    print(logged_line)
    
def log_position_header(seed, strategy, length_table, oc_char='|'):
    filling_ws = ' ' * (length_table - 33)
    logged_line_1 = "{} Seed: {}      Strategy: {}{}{}".format(oc_char, str(seed).zfill(2), strategy, filling_ws, oc_char)
    filling_ws = ' ' * (length_table - 71)
    logged_line_2 = "{} Epoch         Train Accuracy  Test Accuracy   Train Loss   Test Loss{}{}"\
        .format(oc_char, filling_ws, oc_char)
    print(logged_line_1)
    print(logged_line_2)
    
def log_position_line(epoch, num_epochs, train_acc, test_acc, train_loss, test_loss, length_table, oc_char='|'):
    epoch = str(epoch).zfill(len(str(num_epochs))) # leading zeros
    train_acc = str(np.round(train_acc, 2)).zfill(5)
    test_acc = str(np.round(test_acc, 2)).zfill(5)
    train_loss = str(np.round(train_loss, 5)).zfill(7)
    test_loss = str(np.round(test_loss, 5)).zfill(7)
    filling_ws = ' ' * (length_table - 69 )
    logged_line = "{} [{}/{}]:    {}           {}           {}      {}{}{}"\
        .format(oc_char, epoch, num_epochs, train_acc, test_acc, train_loss, test_loss, filling_ws, oc_char)
    print(logged_line)