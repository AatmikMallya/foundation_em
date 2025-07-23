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
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
        
        # Create main flow line
        flow_line = Line(LEFT * 6, RIGHT * 6, color=GRAY_A, stroke_width=2)
        flow_line.shift(DOWN * 0.5)
        self.play(Create(flow_line), run_time=1)
        
        # --- Define positions ---
        input_pos = LEFT * 5.5 + DOWN * 0.5
        patch_pos = LEFT * 2.8 + DOWN * 0.5
        mask_pos = LEFT * 0.3 + DOWN * 0.5
        
        # --- Stage 1: Input Cube ---
        input_cube = Cube(side_length=0.8, fill_opacity=0.3, fill_color=BLUE, stroke_color=WHITE, stroke_width=2)
        input_cube.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        input_cube.move_to(input_pos)
        input_label = Text("Input\n96³", font_size=18, color=WHITE)
        input_label.next_to(input_cube, DOWN, buff=0.3)
        self.play(FadeIn(input_cube), Write(input_label), run_time=1)
        self.wait(0.5)

        # --- Stage 2: Patchify ---
        cubes_per_side = 4
        mini_cube_side = 0.8 / cubes_per_side

        # 2a. Create the subdivided cube (gapless) that will replace the input cube
        subdivision_group = VGroup()
        for i in range(cubes_per_side):
            for j in range(cubes_per_side):
                for k in range(cubes_per_side):
                    c = Cube(side_length=mini_cube_side, fill_opacity=0.7, fill_color=BLUE, stroke_width=0.5, stroke_color=WHITE)
                    offset_x = (i - (cubes_per_side - 1) / 2) * mini_cube_side
                    offset_y = (j - (cubes_per_side - 1) / 2) * mini_cube_side
                    offset_z = (k - (cubes_per_side - 1) / 2) * mini_cube_side
                    c.move_to(np.array([offset_x, offset_y, offset_z]))
                    subdivision_group.add(c)
        
        # Rotate the entire group at once to look like one object
        subdivision_group.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        subdivision_group.move_to(input_pos)

        # 2b. Create the final patch grid (with gaps)
        gap = 0.05
        spacing = mini_cube_side + gap
        patches_group = VGroup()
        for i in range(cubes_per_side):
            for j in range(cubes_per_side):
                for k in range(cubes_per_side):
                    c = Cube(side_length=mini_cube_side, fill_opacity=0.7, fill_color=BLUE, stroke_width=0.5, stroke_color=WHITE)
                    offset_x = (i - (cubes_per_side - 1) / 2) * spacing
                    offset_y = (j - (cubes_per_side - 1) / 2) * spacing
                    offset_z = (k - (cubes_per_side - 1) / 2) * spacing
                    c.move_to(np.array([offset_x, offset_y, offset_z]))
                    patches_group.add(c)
        
        # Apply the same rotation to the final group
        patches_group.rotate(PI/6, axis=UP).rotate(PI/8, axis=RIGHT)
        patches_group.move_to(patch_pos)
        
        patch_label = Text("Patchify\n4³=64 patches", font_size=16, color=WHITE)
        patch_label.next_to(patches_group, DOWN, buff=0.3)
        arrow1 = Arrow(input_cube.get_right(), patches_group.get_left(), buff=0.4, color=WHITE, stroke_width=3)
        
        # Animate the subdivision and move
        self.play(GrowArrow(arrow1))
        self.play(
            ReplacementTransform(input_cube, subdivision_group),
            Write(patch_label),
            run_time=1.5
        )
        self.play(
            Transform(subdivision_group, patches_group),
            run_time=1.5
        )
        self.wait(0.5)

        # --- Stage 3: Masking ---
        # Animate cubes moving to masking position
        self.play(subdivision_group.animate.move_to(mask_pos), run_time=1.5)

        # Apply masking (color change) after moving
        total_patches = len(subdivision_group)
        num_masked = int(total_patches * 0.75)
        masked_indices = random.sample(range(total_patches), num_masked)
        mask_label = Text("Masking\n75% masked", font_size=16, color=WHITE)
        mask_label.next_to(subdivision_group, DOWN, buff=0.3)
        
        arrow2 = Arrow(patch_pos + RIGHT*0.4, mask_pos + LEFT*0.4, buff=0.1, color=WHITE, stroke_width=3)
        self.play(GrowArrow(arrow2))
        self.play(
            *[subdivision_group[idx].animate.set_fill(RED) for idx in masked_indices],
            Write(mask_label),
            run_time=1.5
        )
        self.wait(0.5)
        
        # --- Final cleanup and rest of the animation ---
        # ... (rest of the script can follow) ...
        self.wait(2)
        
        # Stage 4: Encoder
        encoder_pos = RIGHT * 0.5 + DOWN * 0.5
        encoder_rect = RoundedRectangle(
            width=1.2, height=1.5, corner_radius=0.1,
            fill_opacity=0.8, fill_color=ORANGE, stroke_color=ORANGE
        )
        encoder_rect.move_to(encoder_pos)
        encoder_label = Text("ViT Encoder\n12 layers\ndim 768", font_size=14, color=WHITE)
        encoder_label.move_to(encoder_pos)
        
        encoder_title = Text("Encoder", font_size=20, color=ORANGE)
        encoder_title.next_to(encoder_rect, DOWN, buff=0.3)
        
        arrow3 = Arrow(mask_pos + RIGHT*0.3, encoder_pos + LEFT*0.6, buff=0.1, color=WHITE, stroke_width=3)
        
        self.play(
            GrowArrow(arrow3),
            FadeIn(encoder_rect),
            Write(encoder_label),
            Write(encoder_title),
            run_time=1.5
        )
        self.wait(1)
        
        # Transition to decoder section
        self.play(
            FadeOut(VGroup(input_cube, patches_group, arrow1, arrow2, arrow3)),
            run_time=1
        )
        
        # Decoder section title
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