
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

#pretrained_model_name = "siebert/sentiment-roberta-large-english"
class SentimentRegressionModelWithLoRA(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.1):
        super(SentimentRegressionModelWithLoRA, self).__init__()
        
        # Load the pre-trained transformer model
        self.transformer = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Define the regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid to constrain output between 0 and 1
        )
        
        # Apply LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Or SEQUENCE_CLASSIFICATION
            r=8,  # Rank of the update matrices
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",  # Bias parameter
            # Specify which modules to apply LoRA to
            target_modules=["query", "key", "value", "output.dense"]
        )
        
        # Convert the transformer to a LoRA model
        self.transformer = get_peft_model(self.transformer, peft_config)
        
        # Unfreeze LoRA parameters only
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_output = outputs.hidden_states[-1][:, 0, :]
        regression_output = self.regression_head(cls_output)
        return regression_output.squeeze(-1)

class SentimentRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.1):
        super(SentimentRegressionModel, self).__init__()
        
        # Load the pre-trained transformer model
        self.transformer = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Define the regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid to constrain output between 0 and 1
        )
        
        # Freeze the transformer layers
        for param in self.transformer.roberta.parameters():
            param.requires_grad = False
            
        # Only unfreeze the last transformer layer (optional)
        for param in self.transformer.roberta.encoder.layer[-1].parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        # Get the output from the transformer model
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # Use the [CLS] token representation (first token)
        cls_output = outputs.hidden_states[-1][:, 0, :]
        
        # Pass through the regression head
        regression_output = self.regression_head(cls_output)
        
        return regression_output.squeeze(-1)  # Remove last dimension to get shape [batch_size]
    
