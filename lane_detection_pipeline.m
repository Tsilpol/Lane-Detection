function [final_img] = lane_detection_pipeline(img, filter_type)
    % LANE_DETECTION_PIPELINE Δέχεται μια εικόνα και προαιρετικά το είδος του φίλτρου
    % Αν δεν δοθεί το filter_type, χρησιμοποιεί αυτόματα το 'gaussian'
    if nargin < 2
        filter_type = 'gaussian';
    end
    
    %% Μέρος Α: Προεπεξεργασία εικόνας
    % 1. HSV χρωματικός χώρος και Συνδυασμός καναλιών
    if size(img, 3) == 3
        hsv_img = rgb2hsv(img);
        yellow_mask = (hsv_img(:,:,1) < 0.15) & (hsv_img(:,:,2) > 0.4);
        white_mask = (hsv_img(:,:,3) > 0.8) & (hsv_img(:,:,2) < 0.2);
        combined_mask = yellow_mask | white_mask;
        
        gray_img = rgb2gray(img);
        processed_color = gray_img;
        processed_color(~combined_mask) = 0; 
    else
        processed_color = img;
    end
    
    % 2. Εφαρμογή φίλτρων εξομάλυνσης ανάλογα με την επιλογή [cite: 107, 109]
    if strcmp(filter_type, 'median')
        preprocessed_img = medfilt2(processed_color, [5 5]);
    elseif strcmp(filter_type, 'none')
        preprocessed_img = processed_color; % Καμία αποθορυβοποίηση
    else
        preprocessed_img = imgaussfilt(processed_color, 2); % Gaussian
    end
    
    %% Μέρος Β: Δημιουργία περιοχής ενδιαφέροντος (ROI)
    % 1. Βρίσκουμε τις διαστάσεις της εικόνας
    [rows, cols] = size(preprocessed_img);
    
    % 2. Ορίζουμε τις κορυφές του πολυγώνου (τραπέζιο) δυναμικά
    % Συντεταγμένες: [Κάτω-Αριστερά, Κάτω-Δεξιά, Πάνω-Δεξιά, Πάνω-Αριστερά]
    roi_vertices_x = [cols*0.1, cols*0.95, cols*0.55, cols*0.45];
    roi_vertices_y = [rows, rows, rows*0.6, rows*0.6];
    
    % 3. Δημιουργία δυαδικής μάσκας ROI
    roi_mask = poly2mask(roi_vertices_x, roi_vertices_y, rows, cols);
    
    % 4. Εφαρμογή της μάσκας στην προεπεξεργασμένη εικόνα
    roi_img = preprocessed_img;
    roi_img(~roi_mask) = 0; % Μηδενισμός των pixel εκτός ROI
    
   %% Μέρος Γ: Ανίχνευση ακμών
    % Το MATLAB δέχεται τα thresholds του Canny στην κλίμακα [0, 1].
    % Η εκφώνηση προτείνει 50/150 (σε κλίμακα 0-255), άρα διαιρούμε με 255.
    low_thresh = 50 / 255;
    high_thresh = 150 / 255;
    
    % Εφαρμογή του ανιχνευτή Canny στην εικόνα με το ROI
    edges_img = edge(roi_img, 'canny', [low_thresh, high_thresh]);
    
   %% Μέρος Δ: Ανίχνευση ευθειών με μετασχηματισμό Hough
    % 1. Υπολογισμός του μετασχηματισμού Hough
    [H, T, R] = hough(edges_img, 'RhoResolution', 1, 'Theta', -90:0.5:89.5);
    
    % 2. Εντοπισμός κορυφών (Αυξάνουμε τα peaks σε 30, ρίχνουμε το κατώφλι στο 10%)
    P = houghpeaks(H, 30, 'threshold', ceil(0.1*max(H(:))));
    
    % 3. Εξαγωγή ευθύγραμμων τμημάτων (Χαλαρά κριτήρια για διακεκομμένες)
    lines_initial = houghlines(edges_img, T, R, P, 'FillGap', 100, 'MinLength', 15);
    
    % 4. Φιλτράρισμα άκυρων γραμμών (απόρριψη σχεδόν οριζόντιων)
    lines = [];
    for k = 1:length(lines_initial)
        xy = [lines_initial(k).point1; lines_initial(k).point2];
        
        dx = xy(2,1) - xy(1,1);
        dy = xy(2,2) - xy(1,2);
        
        if dx == 0
            slope = inf;
        else
            slope = dy / dx;
        end
        
        % Απορρίπτουμε γραμμές με κλίση κοντά στο μηδέν (π.χ. < 0.3)
        if abs(slope) > 0.3
            lines = [lines, lines_initial(k)];
        end
    end
   %% Μέρος Ε: Εκτίμηση τελικών λωρίδων κυκλοφορίας
    % 1. Ομαδοποίηση υποψήφιων γραμμών 
    left_x = []; left_y = [];
    right_x = []; right_y = [];
    
    for k = 1:length(lines)
        x1 = lines(k).point1(1); y1 = lines(k).point1(2);
        x2 = lines(k).point2(1); y2 = lines(k).point2(2);
        
        slope = (y2 - y1) / (x2 - x1); % Υπολογισμός κλίσης
        
        if slope < -0.3
            left_x = [left_x, x1, x2];
            left_y = [left_y, y1, y2];
        elseif slope > 0.3
            right_x = [right_x, x1, x2];
            right_y = [right_y, y1, y2];
        end
    end
    
    % 2. Εκτίμηση τελικών ευθειών με Γραμμική Παλινδρόμηση (polyfit) 
    y_bottom = size(img, 1);
    y_top = round(size(img, 1) * 0.6); 
    
    poly_pts = []; 
    lines_to_draw = [];
    
    % --- Αριστερή Λωρίδα ---
    if ~isempty(left_x)
        p_left = polyfit(left_y, left_x, 1);
        x_bottom_left = round(polyval(p_left, y_bottom));
        x_top_left = round(polyval(p_left, y_top));
        
        lines_to_draw = [lines_to_draw; x_bottom_left, y_bottom, x_top_left, y_top];
        poly_pts = [poly_pts, x_bottom_left, y_bottom, x_top_left, y_top];
    end
    
    % --- Δεξιά Λωρίδα ---
    if ~isempty(right_x)
        p_right = polyfit(right_y, right_x, 1);
        x_bottom_right = round(polyval(p_right, y_bottom));
        x_top_right = round(polyval(p_right, y_top));
        
        lines_to_draw = [lines_to_draw; x_bottom_right, y_bottom, x_top_right, y_top];
        if length(poly_pts) == 4
            poly_pts = [poly_pts, x_top_right, y_top, x_bottom_right, y_bottom];
        end
    end
    
    % 3. Σχεδιασμός Λωρίδων και Περιοχής 
    % figure('Name', 'Τελικό Αποτέλεσμα Ανίχνευσης');
    imshow(img); hold on;
    
    % Προαιρετικά: Σκίαση εκτιμώμενης περιοχής οδήγησης [cite: 100]
    if length(poly_pts) == 8
        px = [poly_pts(1), poly_pts(3), poly_pts(5), poly_pts(7)];
        py = [poly_pts(2), poly_pts(4), poly_pts(6), poly_pts(8)];
        patch(px, py, 'green', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    
    % Σχεδίαση των τελικών γραμμών [cite: 99]
    for i = 1:size(lines_to_draw, 1)
        plot([lines_to_draw(i,1), lines_to_draw(i,3)], [lines_to_draw(i,2), lines_to_draw(i,4)], 'r', 'LineWidth', 6);
    end
    hold off;
    
    final_img = img; % Επιστρέφουμε την αρχική εικόνα
end