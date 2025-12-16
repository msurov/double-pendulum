import plotly.graph_objects as go
import numpy as np

class CartPendulumAnimator:
    """
    A class to create animations of a cart-pendulum system using Plotly.

    This class handles the drawing and animation logic. It does not simulate
    the dynamics; the user provides the state (cart position x and pendulum
    angle theta) for each frame.
    """
    def __init__(self, cart_width=1.0, cart_height=0.5, pendulum_length=2.0):
        """
        Initializes the animator with the system's physical parameters.

        Args:
            cart_width (float): The width of the cart.
            cart_height (float): The height of the cart.
            pendulum_length (float): The length of the pendulum rod.
        """
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pendulum_length = pendulum_length
        self.frames = []

        # --- Initial Figure Setup ---
        # Create a figure with no initial data, we'll add it in the first frame
        self.fig = go.Figure(
            layout=go.Layout(
                title="Cart-Pendulum System Animation",
                xaxis=dict(range=[-10, 10], zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
                yaxis=dict(range=[-5, 5], zeroline=True, zerolinewidth=2, zerolinecolor='gray', scaleanchor="x", scaleratio=1),
                showlegend=False,
                # Set a fixed aspect ratio so the animation looks correct
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None, {"frame": {"duration": 50, "redraw": True},
                                                             "fromcurrent": True, "transition": {"duration": 10}}]),
                                           dict(label="Pause",
                                                method="animate",
                                                args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                             "mode": "immediate",
                                                             "transition": {"duration": 0}}])])]
            )
        )

    def _calculate_coordinates(self, x, theta):
        """
        Calculates the (x, y) coordinates for all parts of the system.
        """
        # Cart corners (centered at x)
        cart_x = [x - self.cart_width / 2, x + self.cart_width / 2, x + self.cart_width / 2, x - self.cart_width / 2, x - self.cart_width / 2]
        cart_y = [0, 0, self.cart_height, self.cart_height, 0]

        # Pendulum rod and bob
        # Pivot point is at the top center of the cart
        pivot_x = x
        pivot_y = self.cart_height
        
        # Bob position
        bob_x = pivot_x + self.pendulum_length * np.sin(theta)
        bob_y = pivot_y - self.pendulum_length * np.cos(theta)

        return cart_x, cart_y, [pivot_x, bob_x], [pivot_y, bob_y], bob_x, bob_y

    def add_frame(self, x, theta, frame_name=None):
        """
        Adds a single frame to the animation.

        Args:
            x (float): The horizontal position of the cart.
            theta (float): The angle of the pendulum from the vertical (in radians).
            frame_name (str, optional): A name for the frame. Defaults to the frame count.
        """
        if frame_name is None:
            frame_name = str(len(self.frames))
            
        cart_x, cart_y, rod_x, rod_y, bob_x, bob_y = self._calculate_coordinates(x, theta)

        frame_data = [
            # Cart trace
            go.Scatter(x=cart_x, y=cart_y, mode='lines', fill='toself', line=dict(color='blue')),
            # Pendulum rod trace
            go.Scatter(x=rod_x, y=rod_y, mode='lines', line=dict(color='black', width=3)),
            # Pendulum bob trace
            go.Scatter(x=[bob_x], y=[bob_y], mode='markers', marker=dict(size=15, color='red'))
        ]
        
        self.frames.append(go.Frame(data=frame_data, name=frame_name))

    def create_animation(self):
        """
        Assigns the created frames to the figure and adds a slider.
        """
        if not self.frames:
            print("Warning: No frames have been added. Animation will be empty.")
            return

        # Set the initial state of the plot to the first frame
        self.fig.data = self.frames[0].data
        
        # Add all frames to the figure
        self.fig.frames = self.frames

        # Add a slider for navigation
        self.fig.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Frame: "},
                pad={"t": 50},
                steps=[dict(
                    args=[
                        [f.name],
                        {"frame": {"duration": 0, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": 10}}
                    ],
                    label=f.name,
                    method="animate"
                ) for f in self.frames]
            )]
        )

    def save_html(self, filename='cart_pendulum_animation.html'):
        """
        Creates the animation and saves it to an HTML file.

        Args:
            filename (str): The name of the output HTML file.
        """
        self.create_animation()
        self.fig.write_html(filename)
        print(f"Animation saved to {filename}")

def main():
  # --- Example Usage ---

  # 1. Create an instance of the animator
  # You can customize the dimensions here
  animator = CartPendulumAnimator(cart_width=1.5, cart_height=0.6, pendulum_length=2.5)

  # 2. Define the states for each frame of the animation
  # We'll create 100 frames
  num_frames = 100
  for i in range(num_frames):
    # Create some simple, non-physical motion for demonstration
    t = i / 10.0
    
    # Cart position: a sine wave
    x_pos = 5 * np.sin(t)
    
    # Pendulum angle: a cosine wave
    theta_angle = np.pi / 4 * np.cos(2 * t)
    
    # Add this state as a new frame to our animator
    animator.add_frame(x=x_pos, theta=theta_angle)

  # 3. Save the animation to an HTML file
  animator.save_html('my_cart_pendulum.html')

if __name__ == '__main__':
  main()
