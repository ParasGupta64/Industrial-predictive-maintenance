import pytest
from unittest.mock import patch
from llm_explaination import generate_failure_explanation

def test_generate_failure_explanation_prompt_structure():
    """Test that the Gemini prompt includes the strict safety rules."""
    with patch('llm_explaination.genai.Client') as mock_client:
        mock_instance = mock_client.return_value
        mock_response = mock_instance.models.generate_content.return_value
        mock_response.text = "Mock explanation"
        
        result = generate_failure_explanation(
            air_temperature=300.0,
            process_temperature=310.0,
            rotational_speed=1500.0,
            torque=40.0,
            tool_wear=50.0,
            machine_type="L",
            prediction=1
        )
        
        assert result == "Mock explanation"
        
        # Verify the prompt contents
        mock_instance.models.generate_content.assert_called_once()
        args, kwargs = mock_instance.models.generate_content.call_args
        prompt = kwargs.get('contents', '')
        
        # Verify strict rules are still in the prompt
        assert "DO NOT make a new prediction. The ML prediction provided above is FINAL." in prompt
        assert "DO NOT invent numerical safety limits" in prompt
        assert "Treat it only as a categorical machine characteristic" in prompt
        assert "DO NOT claim that a specific component has failed" in prompt

def test_generate_failure_explanation_handles_api_error():
    """Test that the function raises an exception when the API fails."""
    with patch('llm_explaination.genai.Client') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.side_effect = Exception("503 Service Unavailable")
        
        with pytest.raises(Exception) as excinfo:
            generate_failure_explanation(
                air_temperature=300.0,
                process_temperature=310.0,
                rotational_speed=1500.0,
                torque=40.0,
                tool_wear=50.0,
                machine_type="L",
                prediction=1
            )
            
        assert "503 Service Unavailable" in str(excinfo.value)
