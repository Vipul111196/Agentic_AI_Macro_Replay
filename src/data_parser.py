"""
Data Parser for Warmwind OS Conversation Data
Extracts (image, reasoning, action) triplets from conversation JSON files.
"""

import json
import re
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO
from PIL import Image


@dataclass
class ConversationStep:
    """Represents a single step in a conversation workflow."""
    step_index: int
    conversation_id: int
    image_data: Optional[str]  # Base64 encoded image
    think_text: Optional[str]  # Reasoning/thinking text
    action_text: Optional[str]  # Action commands
    raw_assistant_text: Optional[str]  # Full assistant response
    
    def get_image_bytes(self) -> Optional[bytes]:
        """Decode base64 image data to bytes."""
        if self.image_data:
            # Remove data URI prefix if present
            if self.image_data.startswith('data:'):
                # Extract base64 part after comma
                self.image_data = self.image_data.split(',', 1)[1]
            return base64.b64decode(self.image_data)
        return None
    
    def get_pil_image(self) -> Optional[Image.Image]:
        """Convert image data to PIL Image object."""
        img_bytes = self.get_image_bytes()
        if img_bytes:
            return Image.open(BytesIO(img_bytes))
        return None


class StreamingConversationParser:
    """
    Stateful parser that processes messages one at a time (streaming mode).
    
    This simulates real-world conversations where messages arrive incrementally.
    After each message, it checks if conditions are met to create a new step.
    """
    
    def __init__(self):
        self.action_tags = ['Folder', 'click', 'drag', 'focusApp', 'hint', 'moveMouse', 'pressKey', 'releaseKey', 'rightClick',
               'scroll', 'setKeyState', 'setMouseButtonState', 'stop', 'talk', 'type', 'updateWorklist', 'wait']
        
        # State tracking
        self.message_buffer = []  # Buffer of recent messages
        self.step_counter = 0
        self.conversation_id = 0
        self.last_step = None  # Track the most recent step for updating action
        
    def reset(self, conversation_id: int = 0):
        """Reset the parser state for a new conversation."""
        self.message_buffer = []
        self.step_counter = 0
        self.conversation_id = conversation_id
        self.last_step = None
    
    def add_message(self, message: Dict[str, Any]) -> Optional[ConversationStep]:
        """
        Add a new message to the conversation stream.
        
        Returns a NEW ConversationStep only when we see a <think> block.
        Updates the last step's action when we see action tags in subsequent messages.
        
        Pattern:
        1. Assistant message with <think> → CREATE new step (action=None)
        2. Next assistant message with action tags → UPDATE last step's action
        """
        # Add message to buffer
        self.message_buffer.append(message)
        
        # Get current message
        current_msg = self.message_buffer[-1]
        role = current_msg.get('role')
        
        # Only process assistant messages
        if role != 'assistant':
            return None
        
        # Extract text content
        content_list = current_msg.get('content', [])
        current_text = ''
        for content in content_list:
            if content.get('type') == 'text':
                current_text = content.get('text', '')
                break
        
        # Check if current message has <think> block
        think_text = self._extract_think_text(current_text)
        
        if think_text:
            # CASE 1: Message with <think> → CREATE NEW STEP
            
            # Look back for user screenshot (n-1)
            image_data = None
            if len(self.message_buffer) >= 2:
                prev_msg = self.message_buffer[-2]
                if prev_msg.get('role') == 'user':
                    prev_content = prev_msg.get('content', [])
                    for content in prev_content:
                        if content.get('type') == 'image':
                            image_url = content.get('image_url', '')
                            if image_url and 'base64' in image_url:
                                image_data = image_url
                                break
            
            # Create step with action=None (will be updated later if action arrives)
            step = ConversationStep(
                step_index=self.step_counter,
                conversation_id=self.conversation_id,
                image_data=image_data,
                think_text=think_text,
                action_text=None,  # Initially None
                raw_assistant_text=current_text
            )
            
            # Store as last step for potential action update
            self.last_step = step
            self.step_counter += 1
            
            return step
        
        else:
            # CASE 2: Message WITHOUT <think> → Check if it has action tags
            action_text = self._extract_action_text(current_text)
            
            if action_text and self.last_step:
                # Update the last step's action
                self.last_step.action_text = action_text
                # Note: We don't return anything here, just update the existing step
            
            return None
    
    def _extract_think_text(self, text: str) -> Optional[str]:
        """Extract content from <think>...</think> tags."""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_action_text(self, text: str) -> Optional[str]:
        """Extract action commands from various XML-style tags."""
        actions = []
        
        # Find all action tags
        for tag in self.action_tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                actions.append(f'{tag}: {match.strip()}')
        
        return ' | '.join(actions) if actions else None
    
    def process_conversation_streaming(self, messages: List[Dict[str, Any]]) -> List[ConversationStep]:
        """
        Process a full conversation in streaming mode (one message at a time).
        
        This simulates real-world message arrival.
        """
        steps = []
        
        for message in messages:
            step = self.add_message(message)
            if step:
                steps.append(step)
        
        return steps
    
    def parse_file_streaming(self, filepath: str, max_conversations: Optional[int] = None) -> List[ConversationStep]:
        """
        Parse a JSON file using streaming mode.
        
        Each message is processed one at a time, simulating real-world conversation flow.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_steps = []
        conversations_to_process = data[:max_conversations] if max_conversations else data
        
        for conv_idx, conv_obj in enumerate(conversations_to_process):
            self.reset(conversation_id=conv_idx)
            conversation = conv_obj.get('conversation', [])
            steps = self.process_conversation_streaming(conversation)
            all_steps.extend(steps)
        
        return all_steps
    
    def get_statistics(self, steps: List[ConversationStep]) -> Dict[str, Any]:
        """Get statistics about parsed steps."""
        total_steps = len(steps)
        steps_with_images = sum(1 for s in steps if s.image_data)
        steps_with_think = sum(1 for s in steps if s.think_text)
        steps_with_actions = sum(1 for s in steps if s.action_text)
        
        unique_conversations = len(set(s.conversation_id for s in steps))
        
        return {
            'total_steps': total_steps,
            'unique_conversations': unique_conversations,
            'steps_with_images': steps_with_images,
            'steps_with_think': steps_with_think,
            'steps_with_actions': steps_with_actions,
            'avg_steps_per_conversation': total_steps / unique_conversations if unique_conversations > 0 else 0
        }


# Alias for backward compatibility
ConversationParser = StreamingConversationParser


def test_parser_on_sample():
    """Test the streaming parser on a small sample of data."""
    print("=" * 60)
    print("Testing Streaming Parser on Sample Data")
    print("=" * 60)
    
    parser = StreamingConversationParser()
    
    # Test on small sample from training data
    print("\n📂 Loading sample from maf_train.json (streaming mode)...")
    steps = parser.parse_file_streaming(
        'data_validation_split/maf_train.json', 
        max_conversations=1
    )
    
    # Show statistics
    stats = parser.get_statistics(steps)
    print("\n📊 Statistics:")
    print(f"  • Total steps extracted: {stats['total_steps']}")
    print(f"  • Unique conversations: {stats['unique_conversations']}")
    print(f"  • Steps with images: {stats['steps_with_images']}")
    print(f"  • Steps with thinking: {stats['steps_with_think']}")
    print(f"  • Steps with actions: {stats['steps_with_actions']}")
    print(f"  • Avg steps per conversation: {stats['avg_steps_per_conversation']:.2f}")
    
    # Show first few steps in detail
    print("\n🔍 Sample Steps (first 3):")
    for i, step in enumerate(steps[:3]):
        print(f"\n--- Step {i+1} (Conversation {step.conversation_id}, Step {step.step_index}) ---")
        print(f"  Has image: {step.image_data is not None}")
        if step.image_data:
            try:
                img = step.get_pil_image()
                if img:
                    print(f"  Image size: {img.size}")
            except Exception as e:
                print(f"  Image decode error: {e}")
        
        if step.think_text:
            think_preview = step.think_text[:150] + "..." if len(step.think_text) > 150 else step.think_text
            print(f"  Think text: {think_preview}")
        
        if step.action_text:
            print(f"  Actions: {step.action_text}")
    
    print("\n" + "=" * 60)
    print("✅ Parser test complete!")
    print("=" * 60)
    
    return steps


if __name__ == '__main__':
    # Run test when script is executed directly
    test_parser_on_sample()


