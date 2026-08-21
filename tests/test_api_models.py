from wpu_client.models.api import IdentifyRequest, IdentifyResponse, WPULImagesResponse


def test_from_vector_list_joins_floats_as_csv():
    request = IdentifyRequest.from_vector_list(
        type="face", n=4, face_vector=[0.1, 0.2, 0.3], model="sface"
    )

    assert request.type == "face"
    assert request.n == 4
    assert request.model == "sface"
    assert request.face_vector == "0.1,0.2,0.3"


def test_from_vector_list_defaults_to_the_populated_gallery():
    """An omitted `model` must not silently query the empty 128-D gallery."""
    request = IdentifyRequest.from_vector_list(type="face", n=1, face_vector=[1.0])
    assert request.model == "auraface"
    assert IdentifyRequest(type="face", n=1, face_vector="1.0").model == "auraface"


def test_identify_response_success_requires_a_name():
    assert IdentifyResponse(name="Varun").success is True
    assert IdentifyResponse().success is False


def test_identify_response_person_name_falls_back_to_unknown():
    assert IdentifyResponse(name="Varun").person_name == "Varun"
    assert IdentifyResponse().person_name == "Unknown"


def test_wpu_images_response_has_images():
    assert WPULImagesResponse(signed_urls=["https://example.com/a.png"]).has_images is True
    assert WPULImagesResponse().has_images is False
