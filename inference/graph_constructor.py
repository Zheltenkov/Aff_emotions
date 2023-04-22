import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import DataParameters as dp


def update_polar(frame, x, y, scatter):
    dot_1 = x[frame]
    dot_2 = y[frame]
    scatter.set_offsets([dot_1, dot_2])
    return scatter,


def update_lines(frame, x, y1, y2, line1, line2, plot):
    # Update the data of the lines based on the current frame
    line1.set_data(x[:frame + 1], y1[:frame + 1])
    line2.set_data(x[:frame + 1], y2[:frame + 1])
    # Update the data of the plot based on the current frame
    plot.set_data(x[:frame + 1], y1[:frame + 1])
    plot.set_data(x[:frame + 1], y2[:frame + 1])
    # Return the lines and the plot as a sequence of Artist objects
    return line1, line2, plot,


def write_line_animation(y1: list, y2: list, fps: int, num_frames: int):
    x = list(range(1, num_frames + 1))  # Values on the X-axis from 1 to num_frames
    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    line1, = ax.plot([], [], 'b-', label='Valence')
    line2, = ax.plot([], [], 'r-', label='Arousal')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()

    plot, = ax.plot([], [], c='orange')
    plt.xlim(1, num_frames)
    plt.ylim(-1, 1)
    ax.set_xticks([])
    plt.ylabel('Intensity', color='black', fontsize=15, rotation=90, labelpad=10, loc='center')
    plt.xlabel('Frame', color='black', fontsize=15)
    plt.title('Intensity Valence/Arousal', fontsize=15)

    ani = FuncAnimation(fig, update_lines, frames=range(num_frames), blit=True, fargs=(x, y1, y2, line1, line2, plot))
    ani.save(os.path.join(dp.output_video, 'line.mp4'), writer='ffmpeg', fps=fps)


def write_polar_animation(x: list, y: list, fps: int, num_frames: int):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    circle = plt.Circle((0, 0), 1.25, fill=True, color='orange', alpha=0.2)
    circle1 = plt.Circle((0, 0), 1.25, fill=False)
    ax.add_patch(circle)
    ax.add_patch(circle1)

    plt.scatter(-0.15, 0.8, s=50, c='purple', marker='o')
    ax.annotate('Alarmed', (-0.17, 0.9), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.45, 0.82, s=50, c='purple', marker='o')
    ax.annotate('Afraid', (-0.5, 0.92), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.38, 0.7, s=50, c='purple', marker='o')
    ax.annotate('Tense', (-0.3, 0.6), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.65, 0.65, s=50, c='purple', marker='o')
    ax.annotate('Angry', (-0.75, 0.75), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.75, 0.5, s=50, c='purple', marker='o')
    ax.annotate('Frustrated', (-0.95, 0.4), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.73, 0.25, s=50, c='purple', marker='o')
    ax.annotate('Distressed', (-0.95, 0.15), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.6, 0.35, s=50, c='purple', marker='o')
    ax.annotate('Annoyed', (-0.37, 0.35), color='black', ha='center', va='center', size=10)

    plt.scatter(-0.9, -0.1, s=50, c='purple', marker='o')
    ax.annotate('Miserable', (-0.65, -0.1), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.85, -0.4, s=50, c='purple', marker='o')
    ax.annotate('Depressed', (-0.6, -0.45), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.7, -0.3, s=50, c='purple', marker='o')
    ax.annotate('Sad', (-0.55, -0.25), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.79, -0.62, s=50, c='purple', marker='o')
    ax.annotate('Gloomy', (-0.58, -0.62), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.6, -0.9, s=50, c='purple', marker='o')
    ax.annotate('Bored', (-0.6, -0.8), color='black', ha='center', va='center', size=10)
    plt.scatter(-0.25, -1, s=50, c='purple', marker='o')
    ax.annotate('Droopy', (-0.25, -1.1), color='black', ha='center', va='center', size=10)

    plt.scatter(0.1, -0.9, s=50, c='purple', marker='o')
    ax.annotate('Tired', (0.2, -1), color='black', ha='center', va='center', size=10)
    plt.scatter(0.35, -0.95, s=50, c='purple', marker='o')
    ax.annotate('Sleepy', (0.53, -0.95), color='black', ha='center', va='center', size=10)
    plt.scatter(0.9, -0.15, s=50, c='purple', marker='o')
    ax.annotate('Content', (1.05, -0.07), color='black', ha='center', va='center', size=10)
    plt.scatter(0.93, -0.3, s=50, c='purple', marker='o')
    ax.annotate('Satisfied', (0.7, -0.3), color='black', ha='center', va='center', size=10)
    plt.scatter(0.5, -0.45, s=50, c='purple', marker='o')
    ax.annotate('Serene', (0.3, -0.45), color='black', ha='center', va='center', size=10)
    plt.scatter(0.65, -0.45, s=50, c='purple', marker='o')
    ax.annotate('Atease', (0.85, -0.45), color='black', ha='center', va='center', size=10)
    plt.scatter(0.57, -0.55, s=50, c='purple', marker='o')
    ax.annotate('Calm', (0.73, -0.57), color='black', ha='center', va='center', size=10)
    plt.scatter(0.73, -0.73, s=50, c='purple', marker='o')
    ax.annotate('Relaxed', (0.53, -0.73), color='black', ha='center', va='center', size=10)

    plt.scatter(0.15, 1, s=50, c='purple', marker='o')
    ax.annotate('Astonished', (0.42, 1), color='black', ha='center', va='center', size=10)
    plt.scatter(0.55, 0.85, s=50, c='purple', marker='o')
    ax.annotate('Excited', (0.65, 0.75), color='black', ha='center', va='center', size=10)
    plt.scatter(0.25, 0.65, s=50, c='purple', marker='o')
    ax.annotate('Amused', (0.25, 0.55), color='black', ha='center', va='center', size=10)
    plt.scatter(0.8, 0.55, s=50, c='purple', marker='o')
    ax.annotate('Happy', (0.98, 0.55), color='black', ha='center', va='center', size=10)
    plt.scatter(0.65, 0.4, s=50, c='purple', marker='o')
    ax.annotate('Delighted', (0.9, 0.4), color='black', ha='center', va='center', size=10)
    plt.scatter(0.72, 0.22, s=50, c='purple', marker='o')
    ax.annotate('Glad', (0.87, 0.22), color='black', ha='center', va='center', size=10)
    plt.scatter(0.85, 0.1, s=50, c='purple', marker='o')
    ax.annotate('Pleased', (1.05, 0.1), color='black', ha='center', va='center', size=10)

    scatter = plt.scatter(x[0], y[0], s=120, c='orange', marker='o')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(color='black', lw=1)
    plt.axvline(color='black', lw=1)
    plt.xlabel("Mild", fontsize=15, rotation=0)
    plt.ylabel('Unpleasant', fontsize=15, rotation=0, labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2 = ax.twinx()
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.ylabel('Pleasant', color='black', fontsize=15, rotation=0, labelpad=10, loc='center')
    plt.xlabel('Intense', color='black', fontsize=15)
    plt.title('Intense', fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ani = FuncAnimation(fig, update_polar, frames=range(num_frames), blit=True, fargs=(x, y, scatter))
    ani.save(os.path.join(dp.output_video, 'polar.mp4'), writer='ffmpeg', fps=fps)
