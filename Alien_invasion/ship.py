import pygame

class Ship(object):

    def __init__(self, ai_settings, screen):
        self.ai_settings = ai_settings
        self.screen = screen

        # get the ship picture and the rectangles
        self.image = pygame.image.load(ai_settings.ship_image)
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()

        # locate the new ship into the bottom-middle of the screen
        self.rect.centerx = self.screen_rect.centerx
        self.rect.centery = self.screen_rect.centery

        #  use center instead of centerx
        self.center = float(self.rect.centerx)
        self.middle = float(self.rect.centery)

        # move flag
        self.moving_right = False
        self.moving_left = False
        self.moving_up = False
        self.moving_down = False

    def blitme(self):
        # plot the ship
        self.screen.blit(self.image, self.rect)

    def update(self):
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.center += self.ai_settings.move_step
        if self.moving_left and self.rect.left > self.screen_rect.left:
            self.center -= self.ai_settings.move_step
        if self.moving_up and self.rect.top > self.screen_rect.top:
            self.middle -= self.ai_settings.move_step
        if self.moving_down and self.rect.bottom < self.screen_rect.bottom:
            self.middle += self.ai_settings.move_step

        # update the rect
        self.rect.centerx = self.center
        self.rect.centery = self.middle
