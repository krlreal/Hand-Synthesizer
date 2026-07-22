from synthesizer import ChordPlayer
from handtracker import Handtracker

def main():
    chord_player = ChordPlayer()
    handtracker = Handtracker()

    try:
        while True:
            success, img = handtracker.read_frame()
            if not success:
                print("Failed to open camera")
                break

            result = handtracker.process(img)
            handtracker.draw_landmarks(img, result)
            img, root_label, shape_label = handtracker.draw_overlay(img, result)

            if root_label is not None and shape_label is not None:
                new_chord = (root_label, shape_label)
                if new_chord != chord_player.curr_chord:
                    chord_player.play_chord(root_label, shape_label)
            else:
                chord_player.stop()

            handtracker.show_img("Image", img)
            if handtracker.is_quit_pressed("q"):
                break
        
    finally:
        handtracker.release()

if __name__ == "__main__":
    main()