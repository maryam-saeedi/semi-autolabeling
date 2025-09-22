def track(video_name, weight_name, conf_thresh, iou_thresh, mapping, output_stream, running):
    from collections import defaultdict

    model = YOLO(weight_name)
    cap = cv2.VideoCapture(video_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Store the track history
    track_history = defaultdict(lambda: [])

    cnt = -1
    while cap.isOpened() and running.value:
        success, frame = cap.read()
        if not success:
            break
        cnt+=1

        # Run tracking on the frame
        results_ = model.track(
            frame,
            persist=True,
            tracker="parameters.yaml",
            imgsz=1280,
            conf=conf_thresh,
            iou=iou_thresh,
            half=False,
            device=device,
            save=False,
            verbose=False,
            show=False,
            stream=True
        )

        for results in results_:
            results = results.to('cpu')

            # Get boxes and track IDs
            num_boxes = results.boxes.data.shape[0]
            boxes = []
            track_ids = []
            if num_boxes > 0 and results.boxes.id is not None:
                boxes = results.boxes.xywh.cpu().tolist()
                track_ids = results.boxes.id.int().cpu().tolist()

                annotated_frame = results.plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 90:
                        track.pop(0)

                    # Draw track line
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=10)

            for id in track_ids:
                if id not in mapping:
                    mapping[id] = 0

        if cnt%100==0:
            output_stream.put((frame, boxes, track_ids))

    output_stream.put(None)

class App:
    __release_version = "1.0.1.250810"

    def __init__(self) -> None:
        self.step = 0

        w = screen.get_width()
        h = screen.get_height()
        self.w = w
        self.h = h
        offset_x = w//50
        offset_y = h//25
        y = h//20
        
        ## get parameter page
        self.browse_video_btn = Button("Browse Video", w/3-2*(offset_x), y, (2*offset_x, offset_y), func=self.__browse_videos)
        self.video_file_lbl = Label(2*offset_x, offset_y+y, w=w/3-2*offset_x)
        self.video_files_count_lbl = Label(2*offset_x+w/3-2*(offset_x), offset_y, w=w/3-2*offset_x)
        self.browse_output_btn = Button("Browse Output Folder", w/3-2*(offset_x), y, (w-w/3, offset_y), func=self.__browse_output)
        self.output_dir_lbl = Label(w-w/3, offset_y+y, w=w/3-2*offset_x)
        #### ---- box parameteres
        self.box_param_title_lbl = Label(offset_x, 2*h/10, text="Box Prediction Parameters")
        self.load_box_model_btn = Button("Load Box Model", w/6-offset_x, y, (offset_x, 2*h/10+2*offset_y), func=self.__load_box_model)
        self.load_box_model_lbl = Label(offset_x, 2*h/10+2*offset_y+y, w=w/5)
        self.iou_inp = InputBox(w/5+offset_x, 2*h/10+2*offset_y, w/6-offset_x, y, text='0.75')
        self.iou_lbl = Label(w/5+offset_x, 2*h/10+offset_y, text='IoU Threshold:')
        self.conf_inp = InputBox(2*w/5+offset_x, 2*h/10+2*offset_x, w/6-offset_x, y, text='0.2')
        self.conf_lbl = Label(2*w/5+offset_x, 2*h/10+offset_y, text='Box Confidence:')
        #### ---- tracking parameters
        self.track_param_title_lbl = Label(offset_x, 4*h/10, text="Tracking Parameteres")
        self.track_high_thresh_inp = InputBox(offset_x, 4*h/10+2*offset_y,  w/6-offset_x, y, text='0.5')
        self.track_high_thresh_lbl = Label(offset_x, 4*h/10+offset_y, text='Track High Threshold:')
        self.track_low_thresh_inp = InputBox(w/5+offset_x, 4*h/10+2*offset_y,  w/6-offset_x, y, text='0.1')
        self.track_low_thresh_lbl = Label(w/5+offset_x, 4*h/10+offset_y, text='Track Low Threshold:')
        self.track_new_thresh_inp = InputBox(2*w/5+offset_x, 4*h/10+2*offset_y,  w/6-offset_x, y, text='0.5')
        self.track_new_thresh_lbl = Label(2*w/5+offset_x, 4*h/10+offset_y, text='New Track Threshold:')
        self.track_buffer_inp = InputBox(3*w/5+offset_x, 4*h/10+2*offset_y,  w/6-offset_x, y, text='300')
        self.track_buffer_lbl = Label(3*w/5+offset_x, 4*h/10+offset_y, text='Track Buffer Size:')
        self.track_match_thresh_inp = InputBox(4*w/5+offset_x, 4*h/10+2*offset_y,  w/6-offset_x, y, text='0.8')
        self.track_match_thresh_lbl = Label(4*w/5+offset_x, 4*h/10+offset_y, text='Matching Threshold:')
        #### ---- monkey names
        self.monkeys_name_title_lbl = Label(offset_x, 6*h/10, text="Monkey's Names")
        self.monkey_name_lbl = Label(offset_x, 6*h/10+offset_y, text="Monkey Name:")
        self.monkey_name_inp = InputBox(offset_x, 6*h/10+2*offset_y, w/6-offset_x, y, func=self.__enable_btn)
        self.add_monkey_name_btn = Button("Add Name", w/6-offset_x, y, (offset_x,6*h/10+3*offset_y+y), clickable=False, func=self.__add_monkey)
        self.add_monkey_name_hint_lbl = Label(offset_x,6*h/10+3*offset_y+y+y, color=(250,50,100))
        #### --------
        self.process_btn = Button("Process", w/6-offset_x, y, (w-w/6-2*offset_x, h-y-offset_y), clickable=False, func=self.__wait_for_process, process=self.__process)

        self.monkey_list_lst = DropDown(offset_x, offset_y, w/6-offset_x, y, options=["No Label"], enable=False, func=self.__select_monkey)
        self.confirm_btn = Button("Confirm", w/12-offset_x, y, (w-w/12-2*offset_x, h-y-offset_y), func=self.__confirm)
        self.finish_btn = Button("Terminate", w/12-offset_x, y, (offset_x, h-y-offset_y), func=self.__finish)

        self.__init()

    def __init(self):
        self.cuda = torch.cuda.is_available()

        ### waiting page ------
        self.wait = False
        self.convert = False
        ## --------------------

        self.input_video = ''
        self.output_dir = ''
        self.box_weight = None
        self.monkey_list = []

        self.done = False

    def __quit(self):
        try:
            self.tracking_running.value = False
            for i in range(len(self.tracking_on_video_process)):
                self.tracking_on_video_process[i].join()
        except:
            pass

    def __wait_for_process(self, process=None):
        if process is not None:
            self.waiting_process = process

        if not self.wait:
            self.wait = True
            return
        
        if not self.convert:
            self.convert = True
            return
        
        # process()
        self.waiting_process()
        self.wait = False
        self.convert = False
  

    def __prompt_file(self, mode="file", filetype=("all files", "*.*")):
        """Create a Tk file dialog and cleanup when finished"""
        top = tkinter.Tk()
        top.withdraw()  # hide window
        if mode=='file':
            file_name = tkinter.filedialog.askopenfilename(parent=top, filetypes = (filetype,))
        elif mode=='save':
            file_name = tkinter.filedialog.asksaveasfilename(parent=top, filetypes = (filetype,))
        else:
            file_name = tkinter.filedialog.askdirectory(parent=top)
        top.destroy()
        if isinstance(file_name, tuple):
            raise Exception("cancel selection")
        
        return file_name
    
    def __browse_output(self):
        try:
            self.output_dir = self.__prompt_file(mode='folder')
            self.output_dir_lbl.text = self.output_dir
        except:
            self.output_dir_lbl.text = ""

    def __browse_videos(self):
        try:
            folder = self.__prompt_file(mode="folder")
            self.input_video = glob.glob(f'{folder}/*.mp4')+glob.glob(f'{folder}/*.avi')+glob.glob(f'{folder}/*.mkv')
            self.video_file_lbl.text = folder
            if len(self.input_video) > 0:
                self.video_files_count_lbl.text = f"{len(self.input_video)} video(s) found."
                self.video_files_count_lbl.color = (50,20,150)
            else:
                self.video_files_count_lbl.text = f"There is no video in the selected path!"
                self.video_files_count_lbl.color = (150,20,50)
        except Exception as e:
            print(e)
            self.video_file_lbl.text = ""
            self.input_video = []

    def __load_box_model(self):
        try:
            file = self.__prompt_file(filetype=("model weights", "*.engine *.pt") if self.cuda else ("PyTorch models", "*.pt"))
            self.box_weight = file
            self.load_box_model_lbl.text = file
        except:
            self.box_weight = None
            self.load_box_model_lbl.text = ''

    def __enable_btn(self):
        self.add_monkey_name_btn.clickable = len(self.monkey_name_inp.text)

    def __add_monkey(self):
        name = self.monkey_name_inp.text
        if name in self.monkey_list:
            self.add_monkey_name_hint_lbl.text = "Already exists."
        else:
            self.monkey_list.append(name)
            self.add_monkey_name_hint_lbl.text = ""
            self.monkey_name_inp.text = ""
            self.add_monkey_name_btn.clickable = False

    def __process(self):
        self.step += 1

        ## dump tracking parameters -------------
        params = {
            "tracker_type": 'bytetrack',
            "track_high_thresh": self.track_high_thresh_inp.text,
            "track_low_thresh": self.track_low_thresh_inp.text,
            "new_track_thresh": self.track_new_thresh_inp.text,
            "track_buffer": self.track_buffer_inp.text,
            "match_thresh": self.track_match_thresh_inp.text,
            "fuse_score": True
        }
        with open("parameters.yaml", "w", encoding="utf-8") as f:
            yaml.dump(params, f, sort_keys=False)
        ## --------------------------------------

        self.monkey_list_lst.options += self.monkey_list
        self.tracked_videos = []
        self.frame_grid = []
        self.cover_grid = []
        self.color_coded = [(200,200,200)]+[tuple((int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255)))) for _ in range(len(self.monkey_list))]
        self.mapping_ids = [dict()]*len(self.input_video)
        self.tracking_running = Value('b', True)
        self.tracking_on_video_process = []
        for i,video in enumerate(self.input_video):
            self.tracked_videos.append(Queue())
            self.tracking_on_video_process.append(threading.Thread(target = track, args=(video, self.box_weight, float(self.conf_inp.text), float(self.iou_inp.text), self.mapping_ids[i], self.tracked_videos[-1], self.tracking_running)))
            self.tracking_on_video_process[-1].start()
        for i in range(len(self.tracked_videos)):
                frame_info = self.tracked_videos[i].get()
                self.cover_grid.append(ClickableArea(0,0,frame_info[0].shape[1],frame_info[0].shape[0],frame_info[1],{key: self.monkey_list_lst.options[value] for key, value in self.mapping_ids[i].items()}, frame_info[2], {key: self.color_coded[value] for key, value in self.mapping_ids[i].items()}, func=self.__click_on_monkey_box, area_num=i))
                self.frame_grid.append(frame_info)

        # os.makedirs(os.path.join(self.output_dir,'dataset'),exist_ok=True)
        for monkey in self.monkey_list:
            os.makedirs(os.path.join(self.output_dir,'dataset',monkey),exist_ok=True)


    def __click_on_monkey_box(self, item, area_num):
        self.monkey_list_lst.enable = True
        self.monkey_list_lst.draw_menu = True
        self.area_of_interest = area_num
        self.id_of_interest = item

    def __select_monkey(self, selected_option):
        self.monkey_list_lst.enable = False
        self.mapping_ids[self.area_of_interest][self.id_of_interest] = selected_option
        for i in range(len(self.tracked_videos)):
            self.cover_grid[i].mapping = {key: self.monkey_list_lst.options[value] for key, value in self.mapping_ids[i].items()}
            self.cover_grid[i].colors = {key: self.color_coded[value] for key, value in self.mapping_ids[i].items()}
            self.cover_grid[i].update()

    def __confirm(self):
        print("============ confirm ============ ")
        if np.array([self.tracked_videos[i].qsize()>0 for i in range(len(self.tracked_videos))]).any(): 
            ### save confirmed label boxes:
            for i in range(len(self.tracked_videos)):
                frame_info = self.frame_grid[i]
                frame = frame_info[0]
                for rect, id in zip(frame_info[1], frame_info[2]):
                    if self.mapping_ids[i][id]!=0:
                        cropped_box = frame[int(rect[1]-rect[3]/2):int(rect[1]+rect[3]/2),int(rect[0]-rect[2]/2):int(rect[0]+rect[2]/2)]
                        cv2.imwrite(os.path.join(self.output_dir, 'dataset', self.monkey_list_lst.options[self.mapping_ids[i][id]],datetime.now().strftime("%Y%m%d%H%M%S")+'.png'), cropped_box)

            self.frame_grid = []
            self.cover_grid = []
            for i in range(len(self.tracked_videos)):
                frame_info = self.tracked_videos[i].get()
                if frame_info is None:
                    self.done = True
                    break
                self.cover_grid.append(ClickableArea(0,0,frame_info[0].shape[1],frame_info[0].shape[0],frame_info[1],{key: self.monkey_list_lst.options[value] for key, value in self.mapping_ids[i].items()}, frame_info[2], {key: self.color_coded[value] for key, value in self.mapping_ids[i].items()}, func=self.__click_on_monkey_box, area_num=i))
                self.frame_grid.append(frame_info)

    def __finish(self):
        self.__quit()
        self.done = True

    def run(self):
        done = False

        while not done:
            try:
                ## get and handle events in this frame
                self.events = pygame.event.get()
                for event in self.events:
                    if event.type == QUIT:
                        self.__quit()
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self.__quit()
                            pygame.quit()
                            sys.exit()

                screen.fill((255, 255, 255))

                if self.step == 0:  # param page
                    self.process_btn.clickable = len(self.input_video)>0 and (self.box_weight is not None) and self.output_dir!='' and len(self.monkey_list)>0 \
                        and self.track_high_thresh_inp.text!='' and self.track_low_thresh_inp.text!='' and self.track_buffer_inp.text!='' and self.track_match_thresh_inp.text!='' and self.track_new_thresh_inp.text!='' \
                        and self.iou_inp.text!='' and self.conf_inp.text!=''
                    self.browse_video_btn.draw(screen)
                    self.video_file_lbl.draw(screen)
                    self.video_files_count_lbl.draw(screen)
                    self.browse_output_btn.draw(screen)
                    self.output_dir_lbl.draw(screen)
                    self.box_param_title_lbl.draw(screen)
                    self.track_param_title_lbl.draw(screen)
                    self.monkeys_name_title_lbl.draw(screen)
                    #### ---- box parameteres
                    self.load_box_model_btn.draw(screen)
                    self.load_box_model_lbl.draw(screen)
                    self.iou_lbl.draw(screen)
                    self.iou_inp.draw(screen, self.events)
                    self.conf_lbl.draw(screen)
                    self.conf_inp.draw(screen, self.events)
                    #### ---- tracking parameters
                    self.track_high_thresh_lbl.draw(screen)
                    self.track_high_thresh_inp.draw(screen, self.events)
                    self.track_low_thresh_lbl.draw(screen)
                    self.track_low_thresh_inp.draw(screen, self.events)
                    self.track_new_thresh_lbl.draw(screen)
                    self.track_new_thresh_inp.draw(screen, self.events)
                    self.track_buffer_lbl.draw(screen)
                    self.track_buffer_inp.draw(screen, self.events)
                    self.track_match_thresh_lbl.draw(screen)
                    self.track_match_thresh_inp.draw(screen, self.events)
                    #### ---- monkey names
                    self.monkey_name_lbl.draw(screen)
                    self.monkey_name_inp.draw(screen, self.events)
                    self.add_monkey_name_btn.draw(screen)
                    self.add_monkey_name_hint_lbl.draw(screen)
                    for i, name in enumerate(self.monkey_list):
                        lbl = Label(self.w/5+self.w/50+(i//10)*self.w/5, 6*self.h/10+2*self.h/25+(i%5)*self.h/30, w=self.w/20, text=name)
                        lbl.draw(screen)
                    #### ------------
                    self.process_btn.draw(screen)

                elif self.step == 1:   # process page
                    if self.done:
                        font = pygame.font.Font(None, 108)
                        text = font.render("Process is done!", True, (50,50,100))
                        r = text.get_rect()
                        screen.blit(text, ((self.w-r.width)//2, (self.h-r.height)//2-50))
                                
                    else:
                        self.confirm_btn.clickable = np.array([self.tracked_videos[i].qsize()>0 for i in range(len(self.tracked_videos))]).all()
                            
                        self.monkey_list_lst.update(self.events)
                        self.monkey_list_lst.draw(screen)
                        self.confirm_btn.draw(screen)
                        self.finish_btn.draw(screen)

                        n = np.ceil(np.sqrt(len(self.tracked_videos)))
                        h = .9*self.h/n - (n-1)*.01*self.h
                        w = (5/4)*h
                        x = (self.w-(n*w)-((n-1)*.01*self.h))/2     

                        if len(self.frame_grid) == len(self.tracked_videos):
                            for i in range(len(self.tracked_videos)):
                                frame = cv2.resize(self.frame_grid[i][0], (int(w),int(h)))
                                screen.blit(pygame.image.frombuffer(frame.tobytes(), (frame.shape[1],frame.shape[0]), "RGB"), (x+(i%n)*(w+.01*self.h)+self.w//40,.05*self.h+(i//n)*(h+.01*self.h)))
                                self.cover_grid[i].draw(screen, x+(i%n)*(w+.01*self.h)+self.w//40,.05*self.h+(i//n)*(h+.01*self.h),frame.shape[1],frame.shape[0])


                ### transparent waiting page
                if self.wait:
                    s = pygame.Surface((self.w,self.h))  # the size of your rect
                    s.set_alpha(220)                # alpha level
                    s.fill((50,50,50))           # this fills the entire surface
                    screen.blit(s, (0,0)) 
                    font = pygame.font.Font(None, 92)
                    text = font.render("Please Wait ...", True, (255,255,255))
                    r = text.get_rect()
                    screen.blit(text, ((self.w-r.width)//2, (self.h-r.height)//2))
                    self.__wait_for_process()


            except Exception as e:
                traceback.print_exc()

            ### update the screen and limit max framerate
            # real_screen.blit(pygame.transform.scale(screen, real_screen.get_rect().size), (0, 0))

            pygame.display.update()
            mainClock.tick(60)
    

if __name__ == "__main__":
    import cv2
    import pygame
    from pygame.locals import *
    import os
    import glob
    from pathlib import Path
    import gc
    import sys
    import tkinter
    import tkinter.filedialog
    import torch
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from multiprocessing import Process, active_children, freeze_support, Value, Queue, Manager, Lock
    import queue
    import traceback
    from datetime import datetime, timedelta
    import json
    import pandas as pd
    import threading
    import yaml

    pygame.init()
    pygame.display.set_caption('Semi Auto Labeling')
    screen = pygame.display.set_mode((0, 0), (pygame.RESIZABLE), vsync=1) # , pygame.FULLSCREEN
    # screen = real_screen.copy()
    a,b = screen.get_height(), screen.get_width()
    print(screen.get_height(), screen.get_width())
    mainClock = pygame.time.Clock()

    freeze_support()  # for multiprocessing on windows
    torch.multiprocessing.set_start_method('spawn')

    from utils.ui_utils import *
    app = App()
    app.run()

