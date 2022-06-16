import { faPaperPlane } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import React, { useState } from 'react';

import AppService from '../../app.service';
import { BotResponse, UserResponse } from '../../types';
import Chats from '../Chats';
import './Chatbot.scss';

const Chatbot: React.FC = () => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [step, setStep] = useState<number>(0);
  const [editingUserResponse, setEditingUserResponse] = useState<string>('');
  const [botResponse, setBotResponse] = useState<BotResponse>({ message: '' });
  const [submittedUserResponse, setSubmittedUserResponse] = useState<string>('');

  // Set Next Step (for Response and Option Click):
  const setNextStep = (response: string) => {
    setStep((step) => step + 1);
    setSubmittedUserResponse(response);
    let userResponse: UserResponse = {
      utterance: response,
    };
    AppService.postUserResponse(userResponse)
      .then((response: any) => {
        setBotResponse(response.data);
      })
      .catch((e: Error) => {
        console.log(e);
      });
    setEditingUserResponse('');
  };

  const optionClick = (e: React.MouseEvent<HTMLElement>) => {
    let option = e.currentTarget.dataset.id;
    if (option) {
      setNextStep(option);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditingUserResponse(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setNextStep(editingUserResponse);
  };

  return (
    <div className="chat-container">
      <Chats
        userResponse={submittedUserResponse}
        botResponse={botResponse}
        optionClick={optionClick}
      />
      <form onSubmit={(e) => handleSubmit(e)} className="form-container">
        <input onChange={(e) => handleInputChange(e)} value={editingUserResponse}></input>
        <button>
          <FontAwesomeIcon icon={faPaperPlane} />
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
