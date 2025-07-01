from manim import *
import numpy as np
import random

class MAEArchitecture(Scene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#1e1e1e"
        
        # Title
        title = Text("MAE 3D Base-Conv Architecture", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.add(title) # Explicitly add title to the scene for persistence
        self.play(FadeIn(title), run_time=1.5) # Fade in instead of Write
        self.wait(0.5)
        
        # --- Define all static stage objects (initially invisible) ---
        input_pos = LEFT * 5.5 + DOWN * 0.5
        patch_pos = LEFT * 2.8 + DOWN * 0.5
        mask_pos = LEFT * 0.3 + DOWN * 0.5
        flatten_base = mask_pos + RIGHT * 1.2
        encoder_pos = RIGHT * 0.5 + DOWN * 0.5

        # 1. Input Cube (static base)
        input_cube_static = Cube(side_length=0.8, fill_opacity=0.3, fill_color=BLUE, stroke_color=WHITE, stroke_width=3)
        input_cube_static.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        input_cube_static.move_to(input_pos)
        input_cube_static.set_opacity(0) # Initially invisible
        input_label = Text("Input\n96³", font_size=18, color=WHITE)
        input_label.next_to(input_cube_static, DOWN, buff=0.3)
        input_label.set_opacity(0) # Initially invisible

        # 2. Patches at Patchify Stage (static base)
        original_side_length = 0.8
        visible_cubes_per_side = 4
        mini_cube_side = original_side_length / visible_cubes_per_side
        final_gap = 0.05
        final_spacing = mini_cube_side + final_gap
        
        patches_at_patch_pos_static = VGroup()
        for i in range(visible_cubes_per_side):
            for j in range(visible_cubes_per_side):
                for k in range(visible_cubes_per_side):
                    c = Cube(side_length=mini_cube_side, fill_opacity=0.7, fill_color=BLUE, stroke_width=0.2, stroke_color=WHITE)
                    offset_x = (i - (visible_cubes_per_side - 1) / 2) * final_spacing
                    offset_y = (j - (visible_cubes_per_side - 1) / 2) * final_spacing
                    offset_z = (k - (visible_cubes_per_side - 1) / 2) * final_spacing
                    c.move_to(np.array([offset_x, offset_y, offset_z]))
                    patches_at_patch_pos_static.add(c)
        patches_at_patch_pos_static.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        patches_at_patch_pos_static.move_to(patch_pos)
        patches_at_patch_pos_static.set_opacity(0) # Initially invisible

        patch_label = Text("Patchify\n4³=64\npatches", font_size=16, color=WHITE)
        patch_label.next_to(patches_at_patch_pos_static, DOWN, buff=0.3)
        patch_label.set_opacity(0) # Initially invisible

        # 3. Masked Patches at Masking Stage (static base)
        masked_at_mask_pos_static = patches_at_patch_pos_static.copy() # Start from patches appearance
        masked_at_mask_pos_static.move_to(mask_pos)
        masked_at_mask_pos_static.set_opacity(0) # Initially invisible
        
        # Color 75% of this static group red for masking effect
        total_patches_in_group = len(masked_at_mask_pos_static)
        num_masked = int(total_patches_in_group * 0.75)
        masked_indices = random.sample(range(total_patches_in_group), num_masked)
        for idx in masked_indices:
            masked_at_mask_pos_static[idx].set_fill(RED) # Directly set color for static copy

        mask_label = Text("Masking\n75% masked", font_size=16, color=WHITE)
        mask_label.next_to(masked_at_mask_pos_static, DOWN, buff=0.3)
        mask_label.set_opacity(0) # Initially invisible

        # 4. Flattened Patches (static base)
        flattened_patches_static = VGroup()
        vertical_spacing = mini_cube_side * 0.9
        for idx in range(total_patches_in_group):
            c = patches_at_patch_pos_static[idx].copy() # Base copy from original patches for shape
            c.move_to(flatten_base + UP * ((idx - (total_patches_in_group - 1) / 2) * vertical_spacing))
            flattened_patches_static.add(c)
        flattened_patches_static.rotate(PI/2, axis=OUT) # Flatten visually
        flattened_patches_static.set_opacity(0) # Initially invisible

        # --- Setup the scene flow --- (use these static objects for animation targets)
        self.add(input_cube_static, input_label) # Add input static to scene initially
        self.add(patches_at_patch_pos_static, patch_label) # Add static patches
        self.add(masked_at_mask_pos_static, mask_label) # Add static masked patches
        self.add(flattened_patches_static) # Add static flattened patches

        flow_line = Line(LEFT * 6, RIGHT * 6, color=GRAY_A, stroke_width=2)
        flow_line.shift(DOWN * 0.5)
        self.play(Create(flow_line), run_time=1)
        
        # Stage 1: Input Volume - Reveal
        self.play(input_cube_static.animate.set_opacity(1), input_label.animate.set_opacity(1), run_time=1)
        self.wait(0.5)
        
        # Stage 2: Patchify - Animate input_cube transforming into patches_at_patch_pos
        # Create a temporary animated copy of input_cube
        animated_input_to_patches = input_cube_static.copy() # Removed .set_opacity(1)
        
        arrow1 = Arrow(input_cube_static.get_boundary_point(RIGHT), patches_at_patch_pos_static.get_boundary_point(LEFT), buff=0.1, color=WHITE, stroke_width=3)

        self.play(GrowArrow(arrow1), run_time=0.8) # Grow arrow first

        self.play(
            Write(patch_label.set_opacity(1)), # Write label to make it visible
            ReplacementTransform(animated_input_to_patches, patches_at_patch_pos_static), # Transform into static patches
            run_time=2.0
        )
        self.wait(0.5)

        # Stage 3: Masking - Animate patches_at_patch_pos moving and getting masked
        # Create a temporary animated copy of patches_at_patch_pos
        animated_patches_to_mask = patches_at_patch_pos_static.copy() # Removed .set_opacity(1)
        # Removed: self.add(animated_patches_to_mask) here. It will be added by the play().
        
        arrow2 = Arrow(patches_at_patch_pos_static.get_boundary_point(RIGHT), masked_at_mask_pos_static.get_boundary_point(LEFT), buff=0.1, color=WHITE, stroke_width=3)

        self.play(GrowArrow(arrow2), run_time=0.8) # Grow arrow first

        self.play(animated_patches_to_mask.animate.move_to(mask_pos), run_time=1.5) # Then move
        self.wait(0.2)

        # Now, animate the actual masking effect (color change) on the moved copy
        # Then replace the animated copy with the static masked version to ensure persistence
        self.play(
            *[animated_patches_to_mask[idx].animate.set_fill(RED) for idx in masked_indices],
            Write(mask_label.set_opacity(1)), # Write label to make it visible
            ReplacementTransform(animated_patches_to_mask, masked_at_mask_pos_static), # Make static masked cube persistent
            run_time=1.5
        )
        self.wait(0.5)

        # Stage 4: Flattening - Animate masked_patches_to_mask becoming flattened
        # Create a temporary animated copy of the masked patches
        animated_patches_to_flatten = masked_at_mask_pos_static.copy() # Removed .set_opacity(1)
        # Removed: self.add(animated_patches_to_flatten) here.
        
        arrow3 = Arrow(masked_at_mask_pos_static.get_boundary_point(RIGHT), flattened_patches_static.get_boundary_point(LEFT), buff=0.1, color=WHITE, stroke_width=3)
        self.play(GrowArrow(arrow3), run_time=0.8) # Grow arrow first
        self.play(
            Transform(animated_patches_to_flatten, flattened_patches_static), # Transform into flattened static
            run_time=2.0
        )
        self.wait(0.3)
        
        # --- Encoder Stage ---
        encoder_rect = RoundedRectangle(
            width=1.2, height=1.5, corner_radius=0.1,
            fill_opacity=0.8, fill_color=ORANGE, stroke_color=ORANGE
        )
        encoder_rect.move_to(encoder_pos)
        encoder_label = Text("ViT Encoder\n12 layers\ndim 768", font_size=14, color=WHITE)
        encoder_label.move_to(encoder_pos)
        
        encoder_title = Text("Encoder", font_size=20, color=ORANGE)
        encoder_title.next_to(encoder_rect, DOWN, buff=0.3)
        
        arrow_to_encoder = Arrow(flattened_patches_static.get_boundary_point(RIGHT), encoder_rect.get_boundary_point(LEFT), buff=0.1, color=WHITE, stroke_width=3)

        self.play(GrowArrow(arrow_to_encoder), run_time=0.8) # Grow arrow first
        self.play(
            FadeIn(encoder_rect),
            Write(encoder_label),
            Write(encoder_title),
            run_time=1.5
        )
        self.wait(1)
        
        # Clean up temporary animated groups, arrows, and labels
        self.play(
            FadeOut(VGroup(arrow1, arrow2, arrow3, arrow_to_encoder, patch_label, mask_label)),
            FadeOut(animated_input_to_patches), # Just in case it lingered
            FadeOut(animated_patches_to_mask), # Just in case it lingered
            FadeOut(animated_patches_to_flatten), # Just in case it lingered
            run_time=1
        )
        
        # Transition to decoder section (keep previous objects on screen)
        decoder_section_title = Text("Decoder", font_size=24, color=PURPLE)
        decoder_section_title.move_to(UP * 2.5)
        self.play(
            Transform(title, decoder_section_title),
            encoder_rect.animate.move_to(LEFT * 4 + DOWN * 0.5),
            encoder_label.animate.move_to(LEFT * 4 + DOWN * 0.5),
            encoder_title.animate.move_to(LEFT * 4 + DOWN * 2),
            run_time=1.5
        )
        
        # Stage 5: Add Mask Tokens
        tokens_pos = LEFT * 2 + DOWN * 0.5
        
        # Show encoded tokens (green) and mask tokens (gray)
        token_group = VGroup()
        for i in range(6):  # Show more tokens for better representation
            if i < 2:  # Encoded tokens (25%)
                token = Rectangle(width=0.08, height=0.4, fill_opacity=0.8, fill_color=GREEN, stroke_width=0)
            else:  # Mask tokens (75%)
                token = Rectangle(width=0.08, height=0.4, fill_opacity=0.8, fill_color=GRAY, stroke_width=0)
            token_group.add(token)
        
        token_group.arrange(RIGHT, buff=0.03).move_to(tokens_pos)
        token_label = Text("Add mask\ntokens", font_size=16, color=WHITE)
        token_label.next_to(token_group, DOWN, buff=0.3)
        
        arrow4 = Arrow(LEFT * 4 + RIGHT*0.6 + DOWN*0.5, tokens_pos + LEFT*0.3, buff=0.1, color=WHITE, stroke_width=3)
        
        self.play(
            GrowArrow(arrow4),
            FadeIn(token_group),
            Write(token_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Stage 6: ViT Decoder
        vit_decoder_pos = RIGHT * 0.5 + DOWN * 0.5
        vit_decoder_rect = RoundedRectangle(
            width=1.2, height=1.2, corner_radius=0.1,
            fill_opacity=0.8, fill_color=PURPLE, stroke_color=PURPLE
        )
        vit_decoder_rect.move_to(vit_decoder_pos)
        vit_decoder_label = Text("ViT Decoder\n8 layers\ndim 512", font_size=14, color=WHITE)
        vit_decoder_label.move_to(vit_decoder_pos)
        
        vit_decoder_title = Text("ViT Decoder", font_size=16, color=PURPLE)
        vit_decoder_title.next_to(vit_decoder_rect, DOWN, buff=0.3)
        
        arrow5 = Arrow(tokens_pos + RIGHT*0.3, vit_decoder_pos + LEFT*0.6, buff=0.1, color=WHITE, stroke_width=3)
        
        self.play(
            GrowArrow(arrow5),
            FadeIn(vit_decoder_rect),
            Write(vit_decoder_label),
            Write(vit_decoder_title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Stage 7: ConvNeck3D
        conv_pos = RIGHT * 2.5 + DOWN * 0.5
        conv_rect = RoundedRectangle(
            width=1.0, height=1.0, corner_radius=0.1,
            fill_opacity=0.8, fill_color=TEAL, stroke_color=TEAL
        )
        conv_rect.move_to(conv_pos)
        conv_label = Text("ConvNeck3D", font_size=14, color=WHITE)
        conv_label.move_to(conv_pos)
        
        conv_title = Text("3D Conv", font_size=16, color=TEAL)
        conv_title.next_to(conv_rect, DOWN, buff=0.3)
        
        arrow6 = Arrow(vit_decoder_pos + RIGHT*0.6, conv_pos + LEFT*0.5, buff=0.1, color=WHITE, stroke_width=3)
        
        self.play(
            GrowArrow(arrow6),
            FadeIn(conv_rect),
            Write(conv_label),
            Write(conv_title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Stage 8: Unpatchify & Output
        output_pos = RIGHT * 4.5 + DOWN * 0.5
        output_cube = Cube(side_length=0.8, fill_opacity=0.4, fill_color=GREEN, stroke_color=WHITE, stroke_width=3)
        # Rotate individual cube to show 3D while keeping position on straight line
        output_cube.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        output_cube.move_to(output_pos)
        output_label = Text("Reconstructed\nVolume\n96³", font_size=16, color=WHITE)
        output_label.next_to(output_cube, DOWN, buff=0.3)
        
        arrow7 = Arrow(conv_pos + RIGHT*0.5, output_pos + LEFT*0.5, buff=0.1, color=WHITE, stroke_width=3)
        
        self.play(
            GrowArrow(arrow7),
            FadeIn(output_cube),
            Write(output_label),
            run_time=1.5
        )
        self.wait(1)
        
        # Add key information box
        key_info = VGroup(
            Text("Key Features:", font_size=20, color=YELLOW),
            Text("• 75% masking ratio", font_size=16, color=WHITE),
            Text("• 3D convolutional decoder neck", font_size=16, color=WHITE),
            Text("• Spatial structure preservation", font_size=16, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        info_box = SurroundingRectangle(key_info, color=YELLOW, buff=0.3, corner_radius=0.1)
        info_group = VGroup(info_box, key_info)
        info_group.scale(0.8).to_edge(UP + RIGHT, buff=0.5)
        
        self.play(
            FadeIn(info_box),
            Write(key_info),
            run_time=2
        )
        self.wait(2)
        
        # Final highlight of the complete architecture
        all_components = VGroup(
            encoder_rect, encoder_label, encoder_title,
            token_group, token_label,
            vit_decoder_rect, vit_decoder_label, vit_decoder_title,
            conv_rect, conv_label, conv_title,
            output_cube, output_label,
            arrow4, arrow5, arrow6, arrow7, flow_line
        )
        
        self.play(
            all_components.animate.set_stroke(width=3),
            run_time=1
        )
        self.wait(1)
        
        # Fade out everything
        self.play(
            FadeOut(all_components),
            FadeOut(info_group),
            FadeOut(title),
            run_time=2
        )
        self.wait(0.5)