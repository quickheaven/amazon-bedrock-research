# Amazon Bedrock Research
## Amazon Bedrock Learn AI on AWS with Python

### Text Models with Amazon Bedrock
#### Text Generation Parameters
* Max Token Generation Length - shorter max length lead to more focused and concise answer, ideal for straight forward questions. In contrast, a longer max length allows the model to provide more detailed and comprehensive responses.
* Temperature - Controls the randomness or creativity in the model response. A lower temperature (closer to 0) makes the model's response more deterministic and predictable. A higher temperature (closer to 1) increases randomness and sometimes leading to more creating or unexpected outputs. 
* Top P - used to guide the generation of text by the model. Usually developers, only change temperature or top P, but typically not both. Setting a 0 temperature and a very low top P is a good approach to get consistent results.
* Stop or Finish Sequence

### Image Generation
#### Image Generation Parameters
4 key parameters (keep in mind the models do have more unique elements)
* Text Prompt - should be descriptive both in content (what you want in the image) and style (how you want the image presented)
* CFG Scale - control how closely the generated images adhere to input prompt; higher value means model will try harder to generate images that closely match the prompt leading to more precise and relevant images.
* Seed - same consistent random see will allow to generate the same images each time. Good for repeatability.
* Height and Width - Be aware of maximum sizing and Follow the value from model doc.


### RAG-Retrieval Augmented Generation
Allows to retrieve **relevant** information to LL query to help **augment** the text **generation** from the LLM.
