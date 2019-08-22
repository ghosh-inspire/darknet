#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include <sys/file.h>
#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define DEMO 1
#define EDGE_SERVER
#define EDGE_DEVICE

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
#ifdef EDGE_DEVICE
static void * cap;
#endif
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

#define TOP_N_PREDICTIONS (5)
#define CLIENT_NUM_DEVICES (3)
#define CLIENT_SERVER_PORT (65530)
#define IMAGE_DATA_LEN (1024)
#define PREDICTION_THRESHOLD (40)
typedef struct data_t {
    int len;
    float pred_data[TOP_N_PREDICTIONS];
    char id;
} sock_data;

typedef enum {
    FETCH_INFO = 0,
    FETCH_DATA = 1
} fetch_req;

#ifdef EDGE_DEVICE
static float infClient_data[TOP_N_PREDICTIONS];
static pthread_mutex_t lock_client;
#endif

#ifdef EDGE_SERVER
static float server_avg_predictions[CLIENT_NUM_DEVICES];
static pthread_mutex_t lock_server;
static pthread_mutex_t lock_server_file;
static pthread_mutex_t lock_server_init;
#endif

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);
#ifdef EDGE_SERVER
    layer l = net->layers[net->n-1];

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);


    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

#endif
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
#ifdef EDGE_SERVER
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    free_detections(dets, nboxes);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
#endif
#ifdef EDGE_DEVICE
    int top = TOP_N_PREDICTIONS;
    int *indexes = calloc(top, sizeof(int));
    int i = 0;
    float *predictionsn = net->outputn;
    top_k(predictionsn, net->outputsn, top, indexes);

    pthread_mutex_lock(&lock_client);
    for(i = 0; i < top; ++i) {
        int index = indexes[i];
	if(PREDICTION_THRESHOLD < (predictionsn[index] * 100)) {
            printf("cls: %5.2f%%: %s\n", predictionsn[index]*100, demo_names[index]);
	    infClient_data[i] = predictionsn[index];
	} else {
	    infClient_data[i] = 0.0;
	}
    }
    save_image(buff[(buff_index + 2)%3], "inffile1");
    pthread_mutex_unlock(&lock_client);
#endif
    return 0;
}

void *fetch_in_thread(void *ptr)
{
#if 0
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
#endif
    free_image(buff[buff_index]);
#ifdef EDGE_DEVICE
    buff[buff_index] = get_image_from_stream(cap);
#else
    pthread_mutex_lock(&lock_server_file);
    buff[buff_index] = load_image_color("inffile2_server.jpg", 0, 0);
    pthread_mutex_unlock(&lock_server_file);
#endif
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}

void *display_in_thread(void *ptr)
{
    int c = show_image(buff[(buff_index + 1)%3], "Demo", 1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

#ifdef EDGE_SERVER

int server_socket_init(int idx) {
    int server_socket = 0;
    int server_fd = 0;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    struct timeval tv;
       
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        printf("socket failed\n");
	assert(0);
    }
 
    // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt)))
    {
        printf("setsockopt failed\n");
	assert(0);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(CLIENT_SERVER_PORT + idx);
       
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address,
                                 sizeof(address))<0)
    {
        printf("bind failed\n");
	assert(0);
    }

    if (listen(server_fd, 3) < 0)
    {
        printf("listen failed\n");
	assert(0);
    }

    if ((server_socket = accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t*)&addrlen))<0)
    {
        printf("accept failed\n");
	assert(0);
    }

    tv.tv_sec = 2;
    tv.tv_usec = 0;
    if(setsockopt(server_socket, SOL_SOCKET, SO_RCVTIMEO, \
			    (const char*)&tv, sizeof tv)) {
        printf("setsockopt failed\n");
	assert(0);
    }

    printf("client id: %d initialised successfully\n", idx);

    return server_socket;
}

float server_recv(fetch_req state_local, int server_socket) {

    sock_data buffer;
    int ret = 0;
    char server_buff[IMAGE_DATA_LEN + 1];
    int tot = 0;
    int b = 0;
    int i = 0;
    float pred = 0.0;
    FILE* fp = NULL;
    int count = 0;

    switch(state_local) {
       case FETCH_INFO:
	   ret = recv(server_socket, &buffer, sizeof(buffer), MSG_WAITALL);
           assert(ret > 0);
           if (buffer.id != 'I') assert(0);

	   count = 0;
	   for (i = 0; i < TOP_N_PREDICTIONS; i++) {
               pred += buffer.pred_data[i];
	       if(buffer.pred_data[i])
		       count++;
	   }
	   if(count)
	       pred /= count;
#if 0
           printf("pred: %5.2f%%\n", buffer.pred_data[0] * 100);
           printf("pred: %5.2f%%\n", buffer.pred_data[1] * 100);
           printf("pred: %5.2f%%\n", buffer.pred_data[2] * 100);
           printf("pred: %5.2f%%\n", buffer.pred_data[3]* 100);
           printf("pred: %5.2f%%\n", buffer.pred_data[4]* 100);
#endif
           break;
       case FETCH_DATA:
           fp = fopen("inffile1_server.jpg", "wb");
           if(fp != NULL){
               while((b = recv(server_socket, server_buff, sizeof(server_buff) - 1, 0)) > 0) {
                   tot+=b;
                   fwrite(server_buff, 1, b, fp);
		   if(b <= (sizeof(server_buff) - 1)) {
                       if(((unsigned char)server_buff[b-1] == 0xd9) \
		           && ((unsigned char)server_buff[b-2] == 0xff)) {
                           break;
		       }
		   } else {
                       printf("Bad data received b: %d\n", b);
		       assert(0);
		   }
               }
               //printf("Received byte: %d\n",tot);
               if (b < 0) printf("\n------ Receiving error------\n");
               fclose(fp);
           } else {
               error("\n File open error\n");
           }
           break;
       default:
           error("\n Invalid state\n");
           break;
    }
    return pred;
}

void server_send(fetch_req state_local, int server_socket) {

    sock_data buffer;
    int ret = 0;

    switch(state_local) {
       case FETCH_INFO:
           buffer.id = 'I';
	   buffer.len = 0;
	   ret = send(server_socket, &buffer, sizeof(buffer), 0);
           assert(ret > 0);
           break;
       case FETCH_DATA:
           buffer.id = 'D';
	   buffer.len = 0;
	   ret = send(server_socket, &buffer, sizeof(buffer), 0);
           assert(ret > 0);
           break;
       default:
           printf("Invalid state\n");
           assert(0);
           break;
    }
}

void *fetch_server_info_thread(void *ptr) {
    int idx = (int)ptr;
#if 0
    float prev_pred = 0;
    float curr_pred = 0;
#endif
    int server_socket = server_socket_init(idx);

    fetch_req state_local = FETCH_INFO;
    while(1) {
        switch(state_local) {
            case FETCH_INFO:
	        usleep(500*1000);
                pthread_mutex_lock(&lock_server);
	        server_send(FETCH_INFO, server_socket);
#if 0
		curr_pred = server_recv(FETCH_INFO, server_socket);
		server_avg_predictions[idx] = fabsf(curr_pred - prev_pred);
                prev_pred = curr_pred;
#endif
                server_avg_predictions[idx] = server_recv(FETCH_INFO, server_socket);
		if(max_index(server_avg_predictions, CLIENT_NUM_DEVICES) == idx) {
	            state_local = FETCH_DATA;
		} else {
	            state_local = FETCH_INFO;
                    pthread_mutex_unlock(&lock_server);
		}
                break;
            case FETCH_DATA:
	        state_local = FETCH_INFO;
	        server_send(FETCH_DATA, server_socket);
		server_recv(FETCH_DATA, server_socket);
                pthread_mutex_lock(&lock_server_file);
                system("cp inffile1_server.jpg inffile2_server.jpg");
                pthread_mutex_unlock(&lock_server_file);
                pthread_mutex_unlock(&lock_server);
		printf("received data from client id: %d\n", idx);
                break;
            default:
                printf("Invalid state\n");
                assert(0);
                break;
	}
    }
    return 0;
}

#endif

#ifdef EDGE_DEVICE
static int client_socket = 0;
void client_socket_init(void) {

    struct sockaddr_in serv_addr; 
    if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
	assert(0);
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(CLIENT_SERVER_PORT); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
    //if(inet_pton(AF_INET, "192.168.0.1", &serv_addr.sin_addr)<=0)
    { 
        printf("\nInvalid address/ Address not supported \n"); 
	assert(0);
    } 
   
    if (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
    { 
        printf("\nConnection Failed \n"); 
	assert(0);
    } 
}

void client_send(fetch_req state_local) {

    sock_data buffer;
    int ret = 0;
    int i = 0, b = 0;
    FILE *fp = NULL;
    char client_buff[IMAGE_DATA_LEN];

    switch(state_local) {
       case FETCH_INFO:
	   for(i = 0; i < TOP_N_PREDICTIONS; i++) {
	       buffer.pred_data[i] = infClient_data[i];
	   }
           buffer.id = 'I';
	   buffer.len = sizeof(infClient_data) * TOP_N_PREDICTIONS;
	   ret = send(client_socket, &buffer, sizeof(buffer), 0);
	   assert(ret > 0);
           break;
       case FETCH_DATA:
	   fp = fopen("inffile2.jpg", "rb");
           if(fp == NULL) error("\n File open error\n");

           while( (b = fread(client_buff, 1, sizeof(client_buff), fp))>0 ){
               send(client_socket, client_buff, b, 0);
           }
           fclose(fp);
           break;
       default:
           printf("Invalid state\n");
           assert(0);
           break;
    }
}

fetch_req client_recv(void) {

    sock_data buffer;
    int ret = 0;

    ret = recv(client_socket, &buffer, sizeof(buffer), MSG_WAITALL);
    assert(ret > 0);

    if(buffer.id == 'D') {
        return FETCH_DATA;
    } else if (buffer.id == 'I') {
        return FETCH_INFO;
    } else {
	printf("Invalid request from server: %c\n", buffer.id);
        assert(0);
    }
    return FETCH_INFO;
}

void *fetch_client_info_thread(void *ptr) {
    fetch_req state_local = FETCH_INFO;
    client_socket_init();
    while(1) {
        state_local = client_recv();
        switch(state_local) {
            case FETCH_INFO:
                pthread_mutex_lock(&lock_client);
                system("cp inffile1.jpg inffile2.jpg");
		client_send(FETCH_INFO);
                pthread_mutex_unlock(&lock_client);
                break;
            case FETCH_DATA:
                pthread_mutex_lock(&lock_client);
		client_send(FETCH_DATA);
                pthread_mutex_unlock(&lock_client);
                break;
            default:
                printf("Invalid state\n");
                assert(0);
                break;
	}
    }
    return 0;
}
#endif

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    int cnt = 0;
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
#if 0
    pthread_t detect_thread;
    pthread_t fetch_thread;
#endif
    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

#ifdef EDGE_DEVICE
    if(filename){
        printf("video file: %s\n", filename);
        cap = open_video_stream(filename, 0, 0, 0, 0);
    }else{
        cap = open_video_stream(0, cam_index, w, h, frames);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
#endif


#ifdef EDGE_DEVICE
    buff[0] = get_image_from_stream(cap);
#else
    pthread_mutex_lock(&lock_server_file);
    buff[0] = load_image_color("inffile2_server.jpg", 0, 0);
    pthread_mutex_unlock(&lock_server_file);
#endif

    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;
#ifdef EDGE_SERVER
    if(!prefix){
        make_window("Demo", 1352, 1013, fullscreen);
    }
#endif
    demo_time = what_time_is_it_now();

#ifdef EDGE_SERVER
    pthread_t infinfo_server_thread[CLIENT_NUM_DEVICES];
    if (pthread_mutex_init(&lock_server, NULL) != 0) error("\n mutex init has failed\n");
    if (pthread_mutex_init(&lock_server_init, NULL) != 0) error("\n mutex init has failed\n");
    if (pthread_mutex_init(&lock_server_file, NULL) != 0) error("\n mutex init has failed\n");
    for (cnt = 0; cnt < CLIENT_NUM_DEVICES; cnt++) {
        if(pthread_create(&infinfo_server_thread[cnt], 0, \
				fetch_server_info_thread, (void *)cnt))error("\n Thread creation failed\n");
    }
#endif

#ifdef EDGE_DEVICE
    pthread_t infinfo_client_thread;
    if (pthread_mutex_init(&lock_client, NULL) != 0) error("\n mutex init has failed\n");
    if(pthread_create(&infinfo_client_thread, 0, fetch_client_info_thread, 0))error("\n Thread creation failed\n");
#endif

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
#if 0
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
#endif
#if (!defined(EDGE_SERVER) && defined(EDGE_DEVICE))
        for(cnt = 0; cnt < 50; cnt++)
            fetch_in_thread(0);
#endif
#ifdef EDGE_SERVER
	usleep(500*1000);
#endif
        fetch_in_thread(0);
        detect_in_thread(0);
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
#ifdef EDGE_SERVER
            display_in_thread(0);
#endif
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
#if 0
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
#endif
        ++count;
    }

#ifdef EDGE_DEVICE
    pthread_join(infinfo_client_thread, 0);
#endif

#ifdef EDGE_SERVER
    for (cnt = 0; cnt < CLIENT_NUM_DEVICES; cnt++) {
        pthread_join(infinfo_server_thread[cnt], 0);
    }
#endif
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

