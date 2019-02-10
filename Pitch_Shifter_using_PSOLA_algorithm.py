###################################################################################################
# Project name: Pitch_Shifter_using_PSOLA_algorithm
# Description: The program receives a input mono wav file at a sample rate of 44100 samples / sec
#              (can be other sample rate) and writes a new file , at the same sample rate with
#              the pitch shifted.
# Author: Joao Nuno Carvalho
# License: MIT Open Source License
# Note: This program was inspired by the description of the PSOLA algorithm in the book:
#         Speech and Audio Processing: A MATLAB-based Approach
#           by Ian Vince McLoughlin
#
#####################################################################################################

import numpy as np
import wave


def readWAVFilenameToArray(source_WAV_filename):
    # Read file from harddisc.
    wav_handler = wave.open(source_WAV_filename,'rb') # Read only.
    num_frames = wav_handler.getnframes()
    sampl_freq = wav_handler.getframerate()
    wav_frames = wav_handler.readframes(num_frames)

    # Loads the file into a NumPy contigous array.

    # Convert Int16 into float64 in the range of [-1, 1].
    # This means that the sound pressure values are mapped to integer values that can range from -2^15 to (2^15)-1.
    #  We can convert our sound array to floating point values ranging from -1 to 1 as follows.
    signal_temp = np.fromstring(wav_frames, 'Int16')
    signal_array = np.zeros( len(signal_temp), float)

    for i in range(0, len(signal_temp)):
        signal_array[i] = signal_temp[i] / (2.0**15)

    return signal_array, sampl_freq


def writeArrayToWAVFilename(signal_array, sampl_freq, destination_WAV_filename):
    # Converts the NumPy contigous array into frames to be writen into the file.
    # From range [-1, 1] to -/+ 2^15 , 16 bits signed
    signal_temp = np.zeros(len(signal_array), 'Int16')
    for i in range(0, len(signal_temp)):
        signal_temp[i] = int( signal_array[i] * (2.0**15) )

    # Convert float64 into Int16.
    # This means that the sound pressure values are mapped to integer values that can range from -2^15 to (2^15)-1.
    num_frames = signal_temp.tostring()

    # Wrtie file from harddisc.
    wav_handler = wave.open(destination_WAV_filename,'wb') # Write only.
    wav_handler.setnframes(len(signal_array))
    wav_handler.setframerate(sampl_freq)
    wav_handler.setnchannels(1)
    wav_handler.setsampwidth(2) # 2 bytes
    wav_handler.writeframes(num_frames)

def pitch_shift(input_WAV, factor, sample_rate):
    output_WAV = np.zeros(len(input_WAV))
    circular_buffer_length = 256
    circ_buf = np.zeros(circular_buffer_length)
    write_index = 0
    read_index = 0
    write_period = 1.0 / sample_rate
    read_period = write_period * factor
    for i in range(0, len(input_WAV)):
        # Write to the circular buffer.
        # To reduce the audible clicks when the two pointers cross we average the
        # existing value with the new value.
        circ_buf[write_index] = (circ_buf[write_index] + input_WAV[i]) / 2
        #circ_buf[write_index] = input_WAV[i]
        if write_index == circular_buffer_length - 1:
            write_index = 0
        else:
            write_index += 1

        # Calc the next read index on the circular buffer.
        t = i * write_period
        ri = t / read_period
        read_index = int(ri % circular_buffer_length)

        # Read from the circular buffer with different pitch.
        output_WAV[i] = circ_buf[read_index]

    return output_WAV


def ltp(sp):
    """
        lpt     - Long term prediction pitch.
        returns - (B, M)
                   B - Pitch multiplier factor
                   M - Pitch lag/lap
    """
    n = len(sp)
    # upper and lower pitch limits (fs ~ 8KHz- 16KHz).
    pmin = 50
    pmax = 200
    sp2 = sp ** 2  # Pre-calculate the square.
    E = np.zeros(pmax+1) # Creates an array to store the values, not using the first elements.
    for M in range(pmin, pmax + 1):
        e_del = sp[0 : n-M]
        e     = sp[M : n]
        e2    = sp2[M : n]
        E[M]     = np.sum((e_del * e) ** 2) / np.sum(e2)

    # Find M, the optimum pitch period.
    M = np.argmax(E) # Not max value mas Max index.

    # Find B, the pitch gain factor
    e_del = sp[0 : n-M]
    e     = sp[M : n]
    e2    = sp2[M : n]
    B     = np.sum(e_del * e) / sum(e2)

    return (B, M)


def pitch_shift_PSOLA(input_WAV, factor, sample_rate):
    # output_WAV = np.zeros(len(input_WAV))
    sp = input_WAV
    N = len(sp)

    # Determine the pitch with 1 tap LTP.
    B, M = ltp(sp)

    print("B: " + str(B))
    print("M: " + str(M))

    # Scaling ratio.
    sc = factor

    out = np.zeros(len(sp))
    win = np.hamming(int(2*M))
    # Segment the recording into N frames.
    N = int(np.floor(len(sp) / M))

    if sc > 1.0:
        rest = N%3
        N = N - rest
    elif sc <= 1.0:
        if N%2 == 0:
            N = N-1

    # Window each and reconstruct
    for n in range(0, N):
        fr1 = 0 + n*M

        to1 = 0 + n * M + 3 * M
        seg = sp[fr1: to1]
        to2 = 0 + n * M + 2 * M

        max_size = 2*M*sc
        step = sc
        indexes_float = np.arange(0, max_size, step, dtype=np.float64)
        indexes = indexes_float.astype(int)

        seg = seg[indexes]*win
        out[fr1 : to2] = out[fr1 : to2] + seg

    output_WAV = out
    return output_WAV


def change_pitch(input_WAV_path, output_WAV_path, factor):
    input_WAV, sample_rate = readWAVFilenameToArray(input_WAV_path)
    # Pitch shifting.
    output_WAV = pitch_shift_PSOLA(input_WAV, factor, sample_rate)
    # output_WAV = input_WAV.copy()
    writeArrayToWAVFilename(output_WAV, sample_rate, output_WAV_path)



input_WAV_path  = "./Diana_track.wav"

# 1 is the same pitch, 2 half's the pitch and 0.5 duplicates the pitch.

output_WAV_path = "./Diana_track_shifted_0.5_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 0.5 )

output_WAV_path = "./Diana_track_shifted_0.7_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 0.7 )

output_WAV_path = "./Diana_track_shifted_0.8_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 0.8 )

output_WAV_path = "./Diana_track_shifted_1.1_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 1.1 )

output_WAV_path = "./Diana_track_shifted_1.2_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 1.2 )

output_WAV_path = "./Diana_track_shifted_1.3_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 1.3 )

output_WAV_path = "./Diana_track_shifted_1.5_.wav"
change_pitch(input_WAV_path, output_WAV_path, factor = 1.5 )


