interface UserResponse {
  utterance: string;
}

class AppService {
  public async postUserResponse(userResponse: UserResponse) {
    const response = await fetch(`/api/user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userResponse }),
    });
    return await response.json();
  }
}

export default AppService;
