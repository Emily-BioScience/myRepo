class Settings(object):

    def __init__(self):
        # settings for the screen
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (218, 227, 243)
        self.caption = "Alien Invasion"

        # settings for the ship
        self.move_step = 5
        self.ship_image = 'images/rocket.png'

        # settings for the bullet
        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)
        self.bullet_speed = 0.5
        self.bullet_max_num = 3

