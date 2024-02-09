from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np

DT = 0.1
SIM_TICK = 50
NUM_PARTICLES = 25
NUM_LANDMARKS = 8

def main():
    time = 0.0

    df_particles = pd.read_csv('particleData.csv')
    df_RFID = pd.read_csv('Landmark_coords.csv')
    df_history = pd.read_csv('historyData.csv')

    df_RFID.iloc[1:]
    RFID_array = df_RFID.to_numpy()

    euclid_distance_dr = 0.0
    euclid_distance_est = 0.0
    
    while time <= SIM_TICK:
        time+=DT
        low_end = time - 0.005
        high_end = time + 0.005
        selected_row = df_history[(df_history['Time'] >= low_end) & (df_history['Time'] <= high_end)]
        selected_col = selected_row[['hxEst x', 'hxEst y']]

        selected_col.iloc[1:]
        array = selected_col.to_numpy()

        estSubSet = df_history[(df_history['Time'] <= high_end)]
        subSetCols = estSubSet[['hxEst x', 'hxEst y']]
        subSetCols.iloc[1:]

        hxEst = subSetCols.to_numpy()

        xDRSubSet = df_history[(df_history['Time'] <= high_end)]
        subSetCols = xDRSubSet[['hxDr x', 'hxDR y']]
        subSetCols.iloc[1:]

        hxDR = subSetCols.to_numpy()

        xTrueSubSet = df_history[(df_history['Time'] <= high_end)]
        subSetCols = xTrueSubSet[['hxTrue x', 'hxTrue y']]
        subSetCols.iloc[1:]

        hxTrue = subSetCols.to_numpy()

        selected_rows = df_particles[(df_particles['Time'] >= low_end) & (df_particles['Time'] <= high_end)]
        particle_coords = selected_rows[['Particle x', 'Particle y']]
        particle_coords.iloc[1:]
        particles = particle_coords.to_numpy()

        euclid_distance_est += np.linalg.norm(hxTrue - hxEst)**2
        euclid_distance_dr += np.linalg.norm(hxTrue -  hxDR)**2

        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None]
        )
        plt.plot(RFID_array[:,0], RFID_array[:,1], "Dk")

        for i in range(NUM_PARTICLES):
            plt.plot(particles[i,0], particles[i,1], ".r")
            for j in range(1, NUM_LANDMARKS + 1):
                concat_x_string = "Landmark " + str(j) + " x"
                concat_y_string = "Landmark " + str(j) + " y"
                landmark_coords = selected_rows[[concat_x_string, concat_y_string]]
                landmark_coords.iloc[1:]
                landmarks = landmark_coords.to_numpy()
                plt.plot(landmarks[i,0], landmarks[i,1], "xb")

        plt.plot(hxTrue[:,0], hxTrue[:,1], "-b")
        plt.plot(hxDR[:,0], hxDR[:,1], "-g")
        plt.plot(hxEst[:,0], hxEst[:,1], "-r")
        plt.plot(array[0, 0], array[0, 1], "xk")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    euclid_distance_est /= 100
    euclid_distance_dr /= 100

    euclid_distance_est = math.sqrt(euclid_distance_est)
    euclid_distance_dr = math.sqrt(euclid_distance_dr)
    plt.savefig("Simulation Plot Big Circular")
    print(euclid_distance_dr)
    print(euclid_distance_est)

if __name__== "__main__":
    main()   