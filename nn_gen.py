import chess.pgn
import chess
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow import keras



pgn = open("./data/data.pgn")

ldict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}

pdict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

def move_convert(move_string, start=True):
    if start:
        index = (ldict[move_string[0]] * 8) + (int(move_string[1]) - 1)
    else:
        index = (ldict[move_string[2]] * 8) + (int(move_string[3]) - 1)
    out = np.ndarray(shape=(64,))
    out[:] = 0
    out[index] = 1
    return out

def board_convert(board):
    rows = str(board).split('\n')
    out = np.ndarray(shape=(8,8,12))
    for i in range(8):
        out_row = []
        for place in rows[i].split():
            out_row.append(pdict[place])
        out[i] = out_row
    return out

def move_index(move_string):
    index = (ldict[move_string[0]] * 8) + (int(move_string[1]) - 1)
    return index

def examine_data():
    #Headers(Event='?', Site='?', Date='????.??.??', Round='?', White='?', Black='?', Result='*')
    while 1 == 1:
        game = chess.pgn.read_game(pgn)
        x = 1
        print("Event='{}', Site='{}', Date='{}', Round='{}', White='{}', Black='{}', Result='{}'".format(game.headers["Event"], game.headers["Site"], game.headers["Date"], game.headers["Round"], game.headers["White"], game.headers["Black"], game.headers["Result"]))

def generate_data(white=True, sample_size=60000):
    X = np.ndarray(shape=(sample_size, 8, 8, 12))
    Y = np.ndarray(shape=(sample_size, 64))
    Z = np.ndarray(shape=(sample_size, 64))
    sample_count = 0
    while sample_count < sample_size:
        game = chess.pgn.read_game(pgn)
        result = game.headers["Result"]
        if (int(result[0]) == white) and (int(result[2]) == (not white)):
            board = chess.Board()
            move_count = 1
            for move in game.mainline_moves():
                board.push(move)
                if move_count % 2 == white:
                    board_matrix = board_convert(str(board))
                    start_vector = move_convert(str(move), start=True)
                    end_vector = move_convert(str(move), start=False)
                    X[sample_count] = board_matrix
                    Y[sample_count] = start_vector
                    Z[sample_count] = end_vector
                    sample_count += 1
                    if sample_count%1000 == 0:
                        print("generated {} samples".format(sample_count))
                        if sample_count == sample_size:
                            break
                move_count += 1
    return X, Y, Z

def gen_model(train_X, train_Y, directory):
    train_X,valid_X,train_Y,valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)

    #MODEL
    batch_size = 64
    epochs = 20
    num_classes = 64

    chess_model = Sequential()
    chess_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(8, 8, 12)))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D((2, 2), padding='same'))
    chess_model.add(Dropout(0.25))
    chess_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    chess_model.add(Dropout(0.25))
    chess_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    chess_model.add(Dropout(0.4))
    chess_model.add(Flatten())
    chess_model.add(Dense(128, activation='linear'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(Dropout(0.3))
    chess_model.add(Dense(num_classes, activation='softmax'))

    chess_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    chess_model.summary()

    #TRAIN
    chess_model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_Y))
    chess_model.save(directory)
    return



if __name__ == "__main__":

    #examine_data()
    train_X, train_Y , train_Z = generate_data()
    gen_model(train_X, train_Y, './data/start_white')
    gen_model(train_X, train_Z, './data/end_white')


