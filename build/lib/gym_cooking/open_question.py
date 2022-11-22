import pygame
import sys

class Startscreen:
  def __init__(self):
    pygame.init()
    infoObject = pygame.display.Info()
    self.SCREENWIDTH = infoObject.current_w
    self.SCREENHEIGHT = infoObject.current_h
    self.base_font = pygame.font.Font(None, 32)
    self.user_text = ''
    self.color = pygame.Color('chartreuse4')
    self.running = True


  def make_screen(self):
    screen = pygame.display.set_mode([self.SCREENWIDTH, self.SCREENHEIGHT])

    # create rectangle
    input_rect = pygame.Rect(self.SCREENWIDTH / 2, self.SCREENHEIGHT / 2, 140, 32)

    while self.running:
        for event in pygame.event.get():

            # if user types QUIT then the screen will close
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.user_text = self.user_text.strip()
                    if len(self.user_text) > 0:
                        self.running = False
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:

                    # get text input from 0 to -1 i.e. end.
                    self.user_text = self.user_text[:-1]

                # Unicode standard is used for string
                # formation
                else:
                  if event.key != pygame.K_RETURN:
                    self.user_text += event.unicode

        # it will set background color of screen
        screen.fill((255, 255, 255))

        # draw rectangle and argument passed which should
        # be on screen
        pygame.draw.rect(screen, self.color, input_rect)

        text_surface = self.base_font.render(self.user_text, True, (255, 255, 255))

        #enter code text
        entername_text = self.base_font.render("With which strategy did you give advice to the assistant?", True, (0, 0, 0))
        text_rect = entername_text.get_rect(center=(self.SCREENWIDTH / 2, self.SCREENHEIGHT / 2 - 100))
        screen.blit(entername_text, text_rect)

        #Then press enter to begin
        enter_text = self.base_font.render("Press Enter to submit your answer", True, (0, 0, 0))
        text_rect = enter_text.get_rect(center=(self.SCREENWIDTH / 2, self.SCREENHEIGHT / 2 + 100))
        screen.blit(enter_text, text_rect)

        # render at position stated in arguments
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

        # set width of textfield so that text cannot get
        # outside of user's text input

        input_rect.w = max(100, text_surface.get_width() + 10)
        input_rect.x = self.SCREENWIDTH / 2 - input_rect.w/2
        pygame.display.flip()
    return self.user_text
