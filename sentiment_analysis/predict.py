import torch
import argparse
import pandas as pd
from model import TextClassifier
from data_processor import TextProcessor

def predict_sentiment(text, model, processor, device):
    """
    Predict sentiment for a given text
    
    Args:
        text: String text input
        model: Trained model
        processor: Text processor with vocabulary
        device: Device to run inference on
        
    Returns:
        predicted class label and confidence scores
    """
    model.eval()
    
    # Process the text
    sequence = processor.text_to_sequence(text)
    attention_mask = processor.create_attention_mask(sequence)
    
    # Convert to tensors
    input_ids = torch.tensor([sequence], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=1)
        
    # Convert to labels
    labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
    pred_label = labels[prediction.item()]
    
    return pred_label, probs.squeeze().tolist()

def batch_predict_from_csv(file_path, model, processor, device, output_path=None):
    """
    Predict sentiment for all texts in a CSV file
    
    Args:
        file_path: Path to CSV file with a 'text' column
        model: Trained model
        processor: Text processor with vocabulary
        device: Device to run inference on
        output_path: Optional path to save results
        
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(file_path)
    
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")
    
    # Prepare batches
    batch_size = 32
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        sequences = [processor.text_to_sequence(text) for text in batch['text']]
        masks = [processor.create_attention_mask(seq) for seq in sequences]
        
        input_ids = torch.tensor(sequences, dtype=torch.long).to(device)
        attention_mask = torch.tensor(masks, dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            confidence, predictions = torch.max(probs, dim=1)
            
        # Convert to labels
        labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
        pred_labels = [labels[p.item()] for p in predictions]
        
        for j, (idx, row) in enumerate(batch.iterrows()):
            results.append({
                'text': row['text'],
                'predicted_label': pred_labels[j],
                'predicted_class': predictions[j].item(),
                'confidence': confidence[j].item(),
                'negative_prob': probs[j][0].item(),
                'positive_prob': probs[j][1].item(),
                'neutral_prob': probs[j][2].item(),
                'irrelevant_prob': probs[j][3].item(),
            })
    
    results_df = pd.DataFrame(results)
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict sentiment using trained transformer model')
    parser.add_argument('--text', type=str, help='Text to predict sentiment for')
    parser.add_argument('--file', type=str, help='CSV file with texts to predict')
    parser.add_argument('--output', type=str, help='Output file path for batch predictions')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = TextProcessor(max_seq_length=128, min_freq=2)
    
    if not args.text and not args.file:
        print("Please provide either --text or --file argument")
        return
        
    try:
        # Load training data to rebuild vocabulary
        train_data = pd.read_csv("data/twitter_training.csv")
        processor.build_vocabulary(train_data['text'].tolist())
    except Exception as e:
        print(f"Error loading training data: {e}")
        print("Make sure to run this script from the project root directory")
        return
    
    model = TextClassifier(
        vocab_size=len(processor.word2idx),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=128,
        num_classes=4,
        dropout=0.1
    ).to(device)
    
    # Load trained model
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    if args.text:
        label, probs = predict_sentiment(args.text, model, processor, device)
        print(f"\nText: {args.text}")
        print(f"Predicted sentiment: {label}")
        print(f"Confidence scores:")
        print(f"  Negative: {probs[0]:.4f}")
        print(f"  Positive: {probs[1]:.4f}")
        print(f"  Neutral: {probs[2]:.4f}")
        print(f"  Irrelevant: {probs[3]:.4f}\n")
    
    # Batch prediction from file
    if args.file:
        results = batch_predict_from_csv(args.file, model, processor, device, args.output)
        print(f"\nPredicted {len(results)} samples")
        
        # Print sample results
        print("\nSample predictions:")
        for i, row in results.head(5).iterrows():
            print(f"Text: {row['text'][:50]}...")
            print(f"Predicted: {row['predicted_label']} (confidence: {row['confidence']:.4f})\n")

if __name__ == "__main__":
    main()