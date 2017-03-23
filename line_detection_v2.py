import numpy as np
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from can_see_line import Line, calibration, PerspectiveMatrix, abs_sobel_thresh, s_color_thresh, l_color_thresh, dir_thresh, mag_thresh
print("modules loaded!")

mtx, dist = calibration(nx=9, ny=6)
M, Minv = PerspectiveMatrix()
image_size = (1280, 720)
image_count = 0
need_test = 5800
need_plot = need_test - 5

left_line = Line()
right_line = Line()

left_arrow = plt.imread("./left_arrow.jpg")
right_arrow = plt.imread("./right_arrow.jpg")
straight_arrow = plt.imread("./straight.jpg")

def process_image(image):
    global image_count
    image_count = image_count + 1
    print()
    print(image_count)
    left_line.detected = False
    right_line.detected = False
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    ksize=3

    if image_count > need_plot:
        plt.imshow(gray, cmap='gray')
        plt.show()

    sx_binary = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(undist, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_thresh(undist, sobel_kernel=ksize, thresh=(0.7, np.pi/2))
    s_binary = s_color_thresh(undist, thresh=(120, 255))
    l_binary = l_color_thresh(undist, thresh=(120, 255))


    combined_binary = np.zeros_like(sx_binary)
    combined_binary[ ( (s_binary == 1) & (l_binary == 1) ) | (sx_binary == 1) ] = 1

    if image_count > need_plot:
        plt.imshow(combined_binary, cmap='gray')
        plt.show()

    binary_warped = cv2.warpPerspective(combined_binary, M, image_size, flags=cv2.INTER_LINEAR)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    if image_count == 1:
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        leftx_base = left_line.next_x_base
        rightx_base = right_line.next_x_base

    if image_count == need_test:
        print("current leftx_base: ", leftx_base)
        print("current rightx_base: ", rightx_base)

    if image_count == need_test:
        print("leftx_base: ", leftx_base)
        print("rightx_base: ", rightx_base)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        if image_count > need_plot:
            print("window number: ", window)
            print("good_left_inds number: ", len(good_left_inds))
            print("good_right_inds number: ", len(good_right_inds))

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_shift = np.int(np.mean(nonzerox[good_left_inds])) - leftx_current
            if abs(leftx_shift) < 40:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            elif leftx_shift > 40:
                leftx_current = leftx_current + 40
            else:
                leftx_current = leftx_current - 40
        if len(good_right_inds) > minpix:        
            rightx_shift = np.int(np.mean(nonzerox[good_right_inds])) - rightx_current
            if abs(rightx_shift) < 40:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            elif rightx_shift > 40:
                rightx_current = rightx_current + 40
            else:
                rightx_current = rightx_current - 40

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    if (len(left_lane_inds) > 2000) and (len(right_lane_inds) > 2000):
        left_line.detected = True
        right_line.detected = True

    if (left_line.detected == False) or (right_line.detected == False):
        left_fit = left_line.recent_fit_param[-1]
        right_fit = right_line.recent_fit_param[-1]
        left_fitx = left_line.recent_xfitted[-1]
        right_fitx = right_line.recent_xfitted[-1]
    else:
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fit = left_line.fit_param_adjust(left_fit, image_count)
        right_fit = right_line.fit_param_adjust(right_fit, image_count)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        
        left_fitx = left_line.x_adjust(left_fitx, image_count)
        right_fitx = right_line.x_adjust(right_fitx, image_count)

        left_line.next_x_base = int(np.mean(left_fitx[int(len(left_fitx)*0.6):int(len(left_fitx)*0.8)]))
        right_line.next_x_base = int(np.mean(right_fitx[int(len(right_fitx)*0.6):int(len(right_fitx)*0.8)]))

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if image_count > need_plot:
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    left_curverad = left_line.curverad_adjust(left_curverad, image_count)
    right_curverad = right_line.curverad_adjust(right_curverad, image_count)

    left_x_bottem = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
    left_x_top = left_fit[2]
    right_x_bottem = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2]
    right_x_top = right_fit[2]
    line_mid = (right_x_bottem + left_x_bottem)/2
    offset = xm_per_pix*(line_mid - 660) #camera not in the very middle

    # calculate direction and steering
    if left_x_top > left_x_bottem and right_x_top > right_x_bottem:
        left_line.direction = 1 #right
    elif left_x_top < left_x_bottem and right_x_top < right_x_bottem:
        left_line.direction = -1 #left
    else:
        left_line.direction = 100 #bad 
    if left_curverad > 1500 and right_curverad > 1500:
        left_line.direction = 0

    a = [300, 600, 800, 1000, 2000]
    b = [12, 10, 8, 6, 0]
    fit = np.polyfit(a, b, 2)
    curedad = (left_curverad + right_curverad) * 0.5 
    if curedad > 2000:
        curedad = 2000
    if left_line.direction == 1 or left_line.direction == -1:
        steering = left_line.direction * ( abs(fit[0]*curedad**2+fit[1]*curedad+fit[2]) )
        steering = left_line.steering_adjust(steering, image_count)
    elif left_line.direction == 0:
        steering = 0
        #steering = left_line.steering_adjust(steering, image_count)
    elif left_line.direction == 100:
        steering = 0

    print("steering: ", int(steering), "%")

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 140))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    small_left_arrow = cv2.resize(left_arrow, (60,90))
    small_right_arrow = cv2.resize(right_arrow, (60,90))
    small_straight_arrow = cv2.resize(straight_arrow, (60,90))
    if left_line.direction == -1:
        newwarp[500:500+small_left_arrow.shape[0], 620:620+small_left_arrow.shape[1]] = small_left_arrow
    if left_line.direction == 1:
        newwarp[500:500+small_right_arrow.shape[0], 620:620+small_right_arrow.shape[1]] = small_right_arrow
    if left_line.direction == 0 or left_line.direction == 100:
        newwarp[500:500+small_straight_arrow.shape[0], 620:620+small_straight_arrow.shape[1]] = small_straight_arrow
    #plt.imshow(newwarp)
    #plt.show()
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)


    small_size = (320, 180)
    
    index = combined_binary.nonzero()
    a = combined_binary[:,:,np.newaxis]
    aa = np.dstack((a,a,a))
    aa[index[0], index[1]] = [255, 255, 255]
    small_gray = cv2.resize(aa, small_size)

    small_out_img = cv2.resize(out_img, small_size)
    fig, ax = plt.subplots(1, 1)  # create figure & 1 axis
    ax.plot(histogram)
    fig.savefig('./fig.png')
    plt.close(fig) 
    png = Image.open('./fig.png')
    png.load() 
    background = Image.new("RGB", (800,600), (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    background.save('fig.jpg', 'JPEG', quality=80)
    histo = plt.imread('./fig.jpg')
    histo = histo[64:539,102:670]
    small_histo = cv2.resize(histo, (320, 120))

    result[50:50+small_gray.shape[0], 50:50+small_gray.shape[1]] = small_gray
    result[240:240+small_histo.shape[0], 50:50+small_histo.shape[1]] = small_histo
    result[370:370+small_out_img.shape[0], 50:50+small_out_img.shape[1]] = small_out_img
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(result, "image id: " + str(image_count), (800,50), font, 1, (255,255,255), 2)
    if left_curverad < 3000:
        cv2.putText(result, "left curvature: " + "{:.2f}".format(left_curverad)+"m", (800,80), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "left curvature: > 3km", (800,80), font, 1, (255,255,255), 2)
    if right_curverad < 3000:
        cv2.putText(result, "right curvature: " + "{:.2f}".format(right_curverad)+"m", (800,130), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "right curvature: > 3km", (800,130), font, 1, (255,255,255), 2)
    cv2.putText(result, "offset from left: " + "{:.2f}".format(offset)+"m", (800,180), font, 1, (255,255,255), 2)
    #cv2.putText(result, "steering to the right: " + "{:.0f}".format(steering)+"%", (800,230), font, 1, (255,255,255), 2)
    if steering != 0:
        cv2.putText(result, "{:.0f}".format(abs(steering))+"%", (620,490), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "{:.0f}".format(abs(steering))+"%", (630,490), font, 1, (255,255,255), 2)
    #plt.imshow(result)
    #plt.show()
    return result

video_path = './project_video.mp4'
assert os.path.exists(video_path), 'video file not found.'
output = './better_11.mp4'
clip3 = VideoFileClip(video_path)
#output = './P4/hard/challenge_video.mp4'
#clip3 = VideoFileClip('./P4/challenge_video.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(output, audio=False)
