import cv2
import numpy as np

from src.wall_detection.detector import draw_walls
from src.wall_detection.mask_editor import thin_contour, thicken_contour

class ContourProcessor:
    def __init__(self, app):
        self.app = app
    

    def scale_contours_to_original(self, contours, scale_factor):
        """Scale contours back to the original image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour /= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours

    def scale_contours_to_working(self, contours, scale_factor):
        """Scale contours to the working image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour *= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours

    def update_display_from_contours(self):
        """Redraw contours without changing the stored light coordinates."""
        if self.app.current_image is None:
            return
        base_image = self.app.image_processor._get_display_base_image(self.app.current_image)
        contours = self.app.current_contours or []
        if self.app.scale_factor != 1.0 and self.app.original_image is not None:
            contours = self.scale_contours_to_original(contours, self.app.scale_factor)
        self.app.processed_image = draw_walls(base_image, contours)
        if (getattr(self.app, 'current_lights', None)
                and self.app.enable_light_detection.isChecked()):
            from src.wall_detection.light_detector import draw_lights_on_image

            # Positions are already in original-image grid units. Rendering is
            # read-only; repeated redraws must not scale shared dictionaries.
            self.app.processed_image = draw_lights_on_image(
                self.app.processed_image, self.app.current_lights,
                grid_size=70.0, show_range=False, alpha=0.8
            )
        self.app.original_processed_image = self.app.processed_image.copy()
        self.app.refresh_display()

    def delete_selected_contours(self):
        """Delete the selected contours from the current image."""
        if not self.app.selected_contour_indices:
            return
        
        # Save state before modifying
        self.app.mask_processor.save_state()
        
        # Delete selected contours
        for index in sorted(self.app.selected_contour_indices, reverse=True):
            if 0 <= index < len(self.app.current_contours):
                print(f"Deleting contour {index} (area: {cv2.contourArea(self.app.current_contours[index])})")
                self.app.current_contours.pop(index)
        
        # Clear selection and update display
        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

    def thin_selected_contour(self, contour):
        """Thin a single contour using morphological thinning.

        Returns a list of contours (thinning may split one contour into multiple).
        """
        # Create a mask for the contour
        mask = np.zeros(self.app.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Apply the thinning operation using the imported function
        # Pass the current target width and max iterations settings
        thinned_contours = thin_contour(mask, target_width=self.app.target_width, max_iterations=self.app.max_iterations)

        if thinned_contours is not None:
            return thinned_contours
        else:
            return [contour]

    def thin_selected_contours(self):
        """Thin the selected contours."""
        if not self.app.selected_contour_indices:
            return

        # Save state before modifying
        self.app.mask_processor.save_state()

        # Build new contour list, replacing thinned contours (which may split into multiple)
        new_contours = []
        thinned_indices = set(self.app.selected_contour_indices)

        for idx, contour in enumerate(self.app.current_contours):
            if idx in thinned_indices:
                new_contours.extend(self.thin_selected_contour(contour))
            else:
                new_contours.append(contour)

        self.app.current_contours = new_contours

        # Clear selection and update display
        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

    def thicken_selected_contour(self, contour):
        """Thicken a single contour using morphological dilation.

        Returns a list of contours (dilation may merge nearby regions).
        """
        mask = np.zeros(self.app.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        thickened_contours = thicken_contour(mask, target_width=self.app.target_width, max_iterations=self.app.max_iterations)

        if thickened_contours is not None:
            return thickened_contours
        else:
            return [contour]

    def thicken_selected_contours(self):
        """Thicken the selected contours."""
        if not self.app.selected_contour_indices:
            return

        self.app.mask_processor.save_state()

        new_contours = []
        thickened_indices = set(self.app.selected_contour_indices)

        for idx, contour in enumerate(self.app.current_contours):
            if idx in thickened_indices:
                new_contours.extend(self.thicken_selected_contour(contour))
            else:
                new_contours.append(contour)

        self.app.current_contours = new_contours

        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

